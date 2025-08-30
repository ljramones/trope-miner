#!/usr/bin/env python3
"""
Trope Miner — Candidates & Judge
--------------------------------

Utilities to:
  1) Seed quick trope *candidates* via gazetteer matches over chunks
  2) Judge scenes with a local LLM (Ollama) using retrieval + trope definitions
  3) Store findings back into SQLite with evidence spans

CLI
  # Seed candidates for a specific work (fast string/regex matches)
  python trope_miner_tools.py seed-candidates \
    --db ./tropes.db --work-id <WORK_ID>

  # Judge all scenes in a work using retrieval + LLM
  python trope_miner_tools.py judge-scenes \
    --db ./tropes.db \
    --work-id <WORK_ID> \
    --collection trope-miner-nomic-cos \
    --embed-model nomic-embed-text \
    --reasoner-model llama3.1:8b \
    --ollama-url http://localhost:11434 \
    --top-k 4 --threshold 0.55 \
    --trope-collection trope-catalog-nomic-cos \
    --trope-top-k 12

Notes
  • Candidates are *weak signals*; the judge consolidates and scores.
  • Evidence char spans refer to `work.norm_text` offsets.
  • Make sure Chroma and Ollama services are up.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import uuid
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import requests

try:
    import chromadb
except Exception:
    raise SystemExit("chromadb is required: pip install chromadb requests")

# ----------------------------- Small helpers ----------------------------

def get_table_columns(conn: sqlite3.Connection, table: str) -> set:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return {r[1] for r in rows}  # column name at index 1

def ensure_indexes(conn: sqlite3.Connection) -> None:
    """Indexes that are safe to (re)create here without drifting schema."""
    cur = conn.cursor()
    # candidate lookups
    cur.execute("CREATE INDEX IF NOT EXISTS idx_tc_work_scene ON trope_candidate(work_id, scene_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_tc_chunk ON trope_candidate(chunk_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_tc_trope ON trope_candidate(trope_id)")
    # uniqueness guard (matches ingestion.sql)
    cur.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_candidate_span "
        "ON trope_candidate(work_id, trope_id, start, end)"
    )
    # findings lookups + uniqueness (safe no-ops if already present)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_tf_work_scene ON trope_finding(work_id, scene_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_tf_trope ON trope_finding(trope_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_finding_work ON trope_finding(work_id)")
    cur.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_finding_span "
        "ON trope_finding(work_id, trope_id, evidence_start, evidence_end)"
    )
    conn.commit()

# ----------------------------- Embedding & retrieval --------------------

def embed_text(ollama_url: str, model: str, text: str, timeout: int = 120) -> List[float]:
    """Ollama embeddings with compatibility for 'input' and 'prompt'."""
    url = ollama_url.rstrip('/') + "/api/embeddings"
    def _call(payload):
        r = requests.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        emb = data.get("embedding")
        if not emb and isinstance(data.get("data"), list) and data["data"]:
            emb = data["data"][0].get("embedding")
        return emb or []
    emb = _call({"model": model, "input": text}) or _call({"model": model, "prompt": text})
    if not emb:
        raise RuntimeError(f"Empty embedding from {model}")
    return emb

def get_chroma_collection(host: str, port: int, name: str):
    client = chromadb.HttpClient(host=host, port=port)
    try:
        return client.get_collection(name)
    except Exception:
        return client.create_collection(name)

# ----------------------------- Gazetteer candidates --------------------

# Flexible dash class (ASCII + en/em)
_DASH_CLASS = "[-\u2010-\u2015]"

def _escape_piece_allow_dashes(piece: str) -> str:
    """Escape a token safely, then make any dash variant matchable."""
    esc = re.escape(piece)
    # handle raw and escaped unicode dashes that some Pythons might not escape
    for d in ("-", "–", "—", r"\-"):
        esc = esc.replace(d, _DASH_CLASS)
    return esc

def build_pattern(alias: str) -> re.Pattern:
    r"""
    Case-insensitive pattern for an alias:
      - word-boundary-like lookarounds: (?<!\w) ... (?!\w)
      - internal whitespace -> \s+, internal dashes -> [-\u2010-\u2015\s]+
      - simple one-word alphabetic aliases get optional plural (s|es)
    """
    a = alias.strip()
    parts = re.split(r"\s+", a)
    esc = [_escape_piece_allow_dashes(p) for p in parts if p]
    if not esc:
        return re.compile(r"^\b$", re.IGNORECASE)  # never matches

    if len(esc) == 1 and re.fullmatch(r"[A-Za-z]+", parts[0]):
        core = rf"{esc[0]}(?:s|es)?"
    else:
        core = r"(?:[-\u2010-\u2015\s]+)".join(esc)

    return re.compile(rf"(?<!\w){core}(?!\w)", re.IGNORECASE)

@dataclass
class AliasPat:
    trope_id: str
    alias: str    # surface alias (normalized)
    pattern: re.Pattern

def _json_or_legacy_aliases(blob: Optional[str]) -> List[str]:
    """Parse aliases as JSON array; fallback to legacy '|' delimited if needed."""
    if not blob:
        return []
    try:
        data = json.loads(blob)
        if isinstance(data, list):
            return [x for x in data if isinstance(x, str)]
    except Exception:
        pass
    # legacy fallback
    if "|" in blob:
        return [a.strip() for a in blob.split("|") if a.strip()]
    return []

def _norm_alias(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s.strip(",.;:!?\"'()[]{}")

def load_aliases(conn: sqlite3.Connection) -> List[AliasPat]:
    """Load canonical names + JSON aliases, compile robust patterns."""
    rows = conn.execute("SELECT id, name, aliases FROM trope").fetchall()
    out: List[AliasPat] = []
    for tid, name, alias_blob in rows:
        cand = [_norm_alias(name)]
        cand.extend(_norm_alias(a) for a in _json_or_legacy_aliases(alias_blob))
        # stable de-dup
        seen = set()
        for a in cand:
            if not a or a in seen:
                continue
            seen.add(a)
            pat = build_pattern(a)
            out.append(AliasPat(trope_id=tid, alias=a, pattern=pat))
    return out

def seed_candidates(conn: sqlite3.Connection, work_id: str, aliases: List[AliasPat]) -> int:
    """
    Scan chunks for this work; store work-level spans.
    Relies on UNIQUE(work_id,trope_id,start,end) to avoid dupes across runs.
    """
    q = (
        "SELECT id, scene_id, char_start, char_end, text "
        "FROM chunk WHERE work_id=? ORDER BY idx"
    )
    rows = conn.execute(q, (work_id,)).fetchall()
    inserted = 0
    cur = conn.cursor()

    for chunk_id, scene_id, s, e, text in rows:
        if not text:
            continue
        for ap in aliases:
            for m in ap.pattern.finditer(text):
                start = s + m.start()
                end = s + m.end()
                if start < s or end > e:
                    continue
                try:
                    cur.execute(
                        "INSERT OR IGNORE INTO trope_candidate("
                        " id, work_id, scene_id, chunk_id, trope_id, surface, alias, start, end, source, score"
                        ") VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                        (str(uuid.uuid4()), work_id, scene_id, chunk_id,
                         ap.trope_id, m.group(0), ap.alias, start, end, 'gazetteer', 0.5)
                    )
                    if cur.rowcount:
                        inserted += 1
                except sqlite3.IntegrityError:
                    # unique index or FK guard—ignore
                    pass

    conn.commit()
    return inserted

# ----------------------------- Judge with LLM ---------------------------

JUDGE_SYSTEM = (
    "You are a precise trope-mining assistant. "
    "Given a scene, candidate trope names, and their short definitions, "
    "decide which tropes are PRESENT in the scene. Be conservative and evidence-based."
)

JUDGE_INSTRUCTIONS = (
    "Return a JSON array. Each item: {\n"
    "  \"trope_id\": string,            # trope.id from catalog\n"
    "  \"confidence\": number,          # 0..1 calibrated\n"
    "  \"evidence_char_span\": [start,end], # offsets into work.norm_text\n"
    "  \"rationale\": string\n"
    "}\n"
    "Only include tropes that match the scene with confidence >= THRESHOLD."
)

def call_reasoner(ollama_url: str, model: str, prompt: str, temperature: float = 0.2) -> str:
    url = ollama_url.rstrip('/') + "/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": False, "options": {"temperature": temperature}}
    r = requests.post(url, json=payload, timeout=180)
    r.raise_for_status()
    data = r.json()
    return data.get("response", "")

def extract_json(s: str) -> List[dict]:
    # try direct parse, then fenced, then bracket sweep
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, list) else []
    except Exception:
        pass
    m = re.search(r"```json\s*(\[.*?\])\s*```", s, re.S)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    m = re.search(r"(\[\s*{[\s\S]*}\s*\])", s)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    return []

def build_trope_lookup(conn: sqlite3.Connection) -> Tuple[set, Dict[str, str]]:
    rows = conn.execute("SELECT id, name, COALESCE(aliases,'') FROM trope").fetchall()
    idset = set()
    name2id: Dict[str, str] = {}
    for tid, name, aliases_blob in rows:
        idset.add(tid)
        if name:
            name2id[_norm_alias(name)] = tid
        for a in _json_or_legacy_aliases(aliases_blob):
            na = _norm_alias(a)
            if na:
                name2id[na] = tid
    return idset, name2id

def judge_scenes(
    conn: sqlite3.Connection,
    work_id: str,
    collection: str,
    chroma_host: str,
    chroma_port: int,
    embed_model: str,
    reasoner_model: str,
    ollama_url: str,
    top_k: int,
    threshold: float,
    trope_collection: str,
    trope_top_k: int,
) -> int:
    # ensure helpful indexes exist
    ensure_indexes(conn)

    # list scenes for the work (ordered by position)
    scenes = conn.execute(
        "SELECT s.id, s.char_start, s.char_end, w.norm_text "
        "FROM scene s JOIN work w ON w.id = s.work_id WHERE s.work_id=? ORDER BY s.idx",
        (work_id,)
    ).fetchall()

    # load candidate aliases (fallback if no trope_candidate rows yet)
    aliases = load_aliases(conn)
    idset, name2id = build_trope_lookup(conn)

    coll = get_chroma_collection(chroma_host, chroma_port, collection)
    inserted = 0

    # precompute whether 'level' column exists for robust insert
    finding_cols = get_table_columns(conn, "trope_finding")
    has_level = "level" in finding_cols

    for scene_id, s, e, full_text in scenes:
        scene_text = full_text[s:e]

        # candidates: existing gazetteer hits for this scene, else inline detect
        cands = conn.execute(
            "SELECT DISTINCT trope_id FROM trope_candidate WHERE work_id=? AND scene_id=?",
            (work_id, scene_id)
        ).fetchall()
        cand_ids = {row[0] for row in cands}
        if not cand_ids:
            for ap in aliases:
                if ap.pattern.search(scene_text):
                    cand_ids.add(ap.trope_id)

        # retrieval: embed scene, pull top-k supportive chunks
        try:
            q_emb = embed_text(ollama_url, embed_model, scene_text)
        except Exception:
            q_emb = []
        support = []
        if q_emb:
            res = coll.query(
                query_embeddings=[q_emb],
                n_results=top_k,
                include=["metadatas", "distances"],
                where={"work_id": work_id}  # << restrict to this work only
            )
            ids = res.get("ids", [[]])[0]
            metas = res.get("metadatas", [[]])[0]
            for cid, meta in zip(ids, metas):
                # extract excerpt from DB
                row = conn.execute("SELECT work_id, char_start, char_end FROM chunk WHERE id=?", (cid,)).fetchone()
                if not row:
                    continue
                w_id, cs, ce = row
                text = conn.execute(
                    "SELECT substr(norm_text, ?+1, min(400, ?-?)) FROM work WHERE id=?",
                    (cs, ce, cs, w_id)
                ).fetchone()[0] or ""
                support.append(text)

        # semantic shortlist from trope catalog (if provided)
        if q_emb and (not cand_ids or len(cand_ids) < 6):
            try:
                tcoll = get_chroma_collection(chroma_host, chroma_port, trope_collection)
                tres = tcoll.query(query_embeddings=[q_emb], n_results=trope_top_k, include=["metadatas"])
                for tid in tres.get("ids", [[]])[0]:
                    cand_ids.add(tid)
            except Exception:
                pass

        print(f"[debug] scene={scene_id[:8]} cand_count={len(cand_ids)} candidates={sorted(list(cand_ids))[:20]}")

        # build trope definition block
        defs = []
        if cand_ids:
            qmarks = ",".join(["?"] * len(cand_ids))
            rows = conn.execute(
                f"SELECT id, name, summary FROM trope WHERE id IN ({qmarks})",
                tuple(cand_ids)
            ).fetchall()
            for tid, name, summary in rows:
                defs.append(f"- {tid} :: {name} — {summary or ''}")

        print(f"[debug] scene={scene_id[:8]} defs={len(defs)} support_snips={len(support)}")

        # crafting prompt
        avail_ids = sorted(list(cand_ids))
        prompt = (
            f"SYSTEM: {JUDGE_SYSTEM}\n\n"
            f"SCENE (chars {s}-{e}):\n{scene_text[:1200]}\n\n"
            f"CANDIDATE TROPES (id :: name — summary):\n" +
            ("\n".join(defs) if defs else "(none)") + "\n\n"
            f"AVAILABLE_TROPE_IDS (use only these in output):\n" + json.dumps(avail_ids) + "\n\n"
            f"RETRIEVED SUPPORT (snippets):\n" + ("\n---\n".join(support) if support else "(none)") + "\n\n" +
            JUDGE_INSTRUCTIONS.replace("THRESHOLD", str(threshold)) +
            " Also: Use only values from AVAILABLE_TROPE_IDS for 'trope_id'. Do not invent new ids or names.\n"
        )

        resp = call_reasoner(ollama_url, reasoner_model, prompt)
        items = extract_json(resp)
        if not items:
            print(f"[debug] no JSON; model said (first 300 chars): {resp[:300]!r}")
        else:
            try:
                print(f"[debug] scene={scene_id[:8]} items={len(items)} first={json.dumps(items[:1], ensure_ascii=False)}")
            except Exception:
                print(f"[debug] scene={scene_id[:8]} items={len(items)} first=(unserializable)")

        for it in items:
            try:
                tid = it["trope_id"]
                conf = float(it.get("confidence", 0.0))
                if conf < threshold:
                    continue
                # clamp spans to valid range
                span = it.get("evidence_char_span") or [s, min(e, s + len(scene_text))]
                ev_s, ev_e = int(span[0]), int(span[1])
                ev_s = max(0, min(ev_s, len(full_text)))
                ev_e = max(0, min(ev_e, len(full_text)))
                if ev_e < ev_s:
                    ev_s, ev_e = ev_e, ev_s

                rationale = it.get("rationale", "")
                fid = str(uuid.uuid4())

                # schema-aware insert (include 'level' if the column exists)
                if has_level:
                    conn.execute(
                        "INSERT OR REPLACE INTO trope_finding("
                        " id, work_id, scene_id, chunk_id, trope_id, level, confidence,"
                        " evidence_start, evidence_end, rationale, model)"
                        " VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                        (fid, work_id, scene_id, None, tid, "scene", conf, ev_s, ev_e, rationale, reasoner_model)
                    )
                else:
                    conn.execute(
                        "INSERT OR REPLACE INTO trope_finding("
                        " id, work_id, scene_id, chunk_id, trope_id, confidence,"
                        " evidence_start, evidence_end, rationale, model)"
                        " VALUES(?,?,?,?,?,?,?,?,?,?)",
                        (fid, work_id, scene_id, None, tid, conf, ev_s, ev_e, rationale, reasoner_model)
                    )
                inserted += 1
            except Exception as e:
                print(f"[debug] scene={scene_id[:8]} skipping item due to error: {e}; item={it}")
                continue

    conn.commit()
    return inserted

# ----------------------------- CLI -------------------------------------

def build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Trope Miner candidates + judge")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_seed = sub.add_parser("seed-candidates", help="scan chunks for alias matches and store candidates")
    p_seed.add_argument("--db", required=True)
    p_seed.add_argument("--work-id", required=True)

    p_j = sub.add_parser("judge-scenes", help="LLM-based scene judgments with retrieval")
    p_j.add_argument("--trope-collection", default="trope-catalog-nomic-cos")
    p_j.add_argument("--trope-top-k", type=int, default=12)
    p_j.add_argument("--db", required=True)
    p_j.add_argument("--work-id", required=True)
    p_j.add_argument("--collection", required=True)
    p_j.add_argument("--chroma-host", default=os.getenv("CHROMA_HOST", "localhost"))
    p_j.add_argument("--chroma-port", type=int, default=int(os.getenv("CHROMA_PORT", "8000")))
    p_j.add_argument("--embed-model", default="nomic-embed-text")
    p_j.add_argument("--reasoner-model", default="llama3.1:8b")
    p_j.add_argument("--ollama-url", default=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    p_j.add_argument("--top-k", type=int, default=4)
    p_j.add_argument("--threshold", type=float, default=0.55)

    return p

def main():
    ap = build_cli()
    args = ap.parse_args()
    conn = sqlite3.connect(args.db)
    ensure_indexes(conn)

    if args.cmd == "seed-candidates":
        aliases = load_aliases(conn)
        n = seed_candidates(conn, args.work_id, aliases)
        print(f"Seeded {n} candidate hits for work {args.work_id}")

    elif args.cmd == "judge-scenes":
        n = judge_scenes(
            conn=conn,
            work_id=args.work_id,
            collection=args.collection,
            chroma_host=args.chroma_host,
            chroma_port=args.chroma_port,
            embed_model=args.embed_model,
            reasoner_model=args.reasoner_model,
            ollama_url=args.ollama_url,
            top_k=args.top_k,
            threshold=args.threshold,
            trope_collection=args.trope_collection,
            trope_top_k=args.trope_top_k,
        )
        print(f"Inserted {n} trope findings for work {args.work_id}")

if __name__ == "__main__":
    main()
