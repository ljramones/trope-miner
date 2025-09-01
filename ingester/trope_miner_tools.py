#!/usr/bin/env python3
"""
Trope Miner — Candidates & Judge (with two-stage rerank + sanity)
-----------------------------------------------------------------

Utilities to:
  1) Seed quick trope *candidates* via gazetteer matches over chunks
  2) Judge scenes with a local LLM (Ollama) using:
       - per-work retrieval from Chroma,
       - LLM rerank (choose top 2–3 support snippets),
       - trope shortlist sanity (lexical + semantic down-weight)
  3) Store findings back into SQLite with evidence spans

CLI
  # Seed candidates for a specific work (fast string/regex matches)
  python trope_miner_tools.py seed-candidates \
    --db ./tropes.db --work-id <WORK_ID>

  # Judge all scenes in a work using retrieval + LLM rerank + sanity
  python trope_miner_tools.py judge-scenes \
    --db ./tropes.db \
    --work-id <WORK_ID> \
    --collection trope-miner-v1-cos \
    --embed-model nomic-embed-text \
    --reasoner-model llama3.1:8b \
    --ollama-url http://localhost:11434 \
    --top-k 8 --threshold 0.55 \
    --trope-collection trope-catalog-nomic-cos \
    --trope-top-k 16
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

# two-stage retrieval + sanity (implemented in rerank_support.py)
from rerank_support import choose_support_and_sanity

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
    alias: str
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
        seen = set()
        for a in cand:
            if not a or a in seen:
                continue
            seen.add(a)
            out.append(AliasPat(trope_id=tid, alias=a, pattern=build_pattern(a)))
    return out


def seed_candidates(conn: sqlite3.Connection, work_id: str, aliases: List[AliasPat]) -> int:
    """
    Scan chunks for this work; store work-level spans.
    Relies on UNIQUE(work_id,trope_id,start,end) to avoid dupes across runs.
    """
    q = ("SELECT id, scene_id, char_start, char_end, text "
         "FROM chunk WHERE work_id=? ORDER BY idx")
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
    "  \"trope_id\": string,                  # trope.id from catalog\n"
    "  \"confidence\": number,                # 0..1 calibrated\n"
    "  \"evidence_char_span\": [start,end],   # offsets into work.norm_text\n"
    "  \"rationale\": string\n"
    "}\n"
    "Only include tropes that match the scene with confidence >= THRESHOLD."
)


def call_reasoner(ollama_url: str, model: str, prompt: str, temperature: float = 0.2) -> str:
    url = ollama_url.rstrip('/') + "/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": False, "options": {"temperature": temperature}}
    r = requests.post(url, json=payload, timeout=180)
    r.raise_for_status()
    return r.json().get("response", "")


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


def _wire_env_for_rerank(
    *,
    chroma_host: str,
    chroma_port: int,
    chunk_collection: str,
    embed_model: str,
    reasoner_model: str,
    ollama_url: str,
    top_k: int,
    keep_m_default: int = 3,
) -> None:
    """
    Ensure rerank_support.py sees the same runtime config via env.
    This avoids drift between CLI args here and env reads there.
    """
    os.environ["CHROMA_HOST"] = chroma_host
    os.environ["CHROMA_PORT"] = str(chroma_port)
    os.environ["CHUNK_COLLECTION"] = chunk_collection
    os.environ["EMBED_MODEL"] = embed_model
    os.environ["REASONER_MODEL"] = reasoner_model
    os.environ["OLLAMA_BASE_URL"] = ollama_url

    # Pass Stage-1/Stage-2 knobs
    os.environ["RERANK_TOP_K"] = str(top_k)
    # Respect existing RERANK_KEEP_M if explicitly set; else default
    os.environ.setdefault("RERANK_KEEP_M", str(keep_m_default))


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
    """
    For each scene in the work:
      - Build a candidate trope list (gazetteer + semantic shortlist).
      - Choose 2–3 support snippets via per-work KNN → LLM rerank.
      - Compute sanity weights (lexical + semantic) BEFORE judging.
      - Prompt the LLM with scene + support + candidate defs + prior weights.
      - Insert findings using adjusted confidence: adj = raw * weight.
    """
    ensure_indexes(conn)

    # Make sure the reranker sees the same runtime settings
    _wire_env_for_rerank(
        chroma_host=chroma_host,
        chroma_port=chroma_port,
        chunk_collection=collection,
        embed_model=embed_model,
        reasoner_model=reasoner_model,
        ollama_url=ollama_url,
        top_k=top_k,
        keep_m_default=3,
    )

    # list scenes (ordered)
    scenes = conn.execute(
        "SELECT s.id, s.idx, s.char_start, s.char_end, w.norm_text "
        "FROM scene s JOIN work w ON w.id = s.work_id "
        "WHERE s.work_id=? ORDER BY s.idx",
        (work_id,)
    ).fetchall()

    # candidate aliases (fallback if no trope_candidate rows yet)
    aliases = load_aliases(conn)
    _idset, _name2id = build_trope_lookup(conn)

    inserted = 0

    # schema probe
    finding_cols = get_table_columns(conn, "trope_finding")
    has_level = "level" in finding_cols

    # max chars used when embedding a scene for the trope catalog query
    SEM_EMBED_MAX = int(os.getenv("SEM_EMBED_MAX_CHARS", "4000"))

    for scene_id, idx, s, e, full_text in scenes:
        scene_text = full_text[s:e]

        # --- 1) Candidate shortlist (gazetteer) ----------------------------
        cands = conn.execute(
            "SELECT DISTINCT trope_id FROM trope_candidate WHERE work_id=? AND scene_id=?",
            (work_id, scene_id)
        ).fetchall()
        cand_ids = {row[0] for row in cands}
        if not cand_ids:  # fallback lexical scan in-scene
            for ap in aliases:
                if ap.pattern.search(scene_text):
                    cand_ids.add(ap.trope_id)

        # --- 2) Semantic shortlist from trope catalog (robust) -------------
        scene_for_sem = scene_text if len(scene_text) <= SEM_EMBED_MAX else scene_text[:SEM_EMBED_MAX]
        q_emb = None
        try:
            q_emb = embed_text(ollama_url, embed_model, scene_for_sem)
        except Exception as ex:
            print(f"[judge] warn: catalog embed failed (len={len(scene_text)} trimmed={len(scene_for_sem)}): {ex}")

        if q_emb:
            try:
                tcoll = get_chroma_collection(chroma_host, chroma_port, trope_collection)
                # do NOT request "ids" in include; Chroma returns ids by default
                tres = tcoll.query(query_embeddings=[q_emb], n_results=trope_top_k, include=["metadatas"])
                ids_from_catalog = (tres.get("ids") or [[]])[0] or []
                for tid in ids_from_catalog:
                    if tid:
                        cand_ids.add(tid)
            except Exception as ex:
                print(f"[judge] warn: catalog query failed: {ex}")

        avail_ids = sorted(list(cand_ids))
        print(f"[judge] scene={scene_id[:8]} cand_after_catalog={len(avail_ids)}")
        if not avail_ids:
            continue  # nothing to judge in this scene

        # --- 3) Two-stage rerank + sanity ---------------------------------
        support_ids, weights = choose_support_and_sanity(
            conn=conn,
            work_id=work_id,
            scene_id=scene_id,
            scene_text=scene_text,
            candidate_trope_ids=avail_ids,
            persist=True,
        )

        # Fetch chosen support texts (prefer chunk.text; fallback to work slice)
        support_texts: List[str] = []
        if support_ids:
            q = conn.execute(
                f"SELECT c.id, c.text, c.char_start, c.char_end, c.work_id "
                f"FROM chunk c WHERE c.id IN ({','.join(['?']*len(support_ids))})",
                tuple(support_ids)
            ).fetchall()
            by_id = {row[0]: row for row in q}
            for cid in support_ids:
                r = by_id.get(cid)
                if not r:
                    continue
                txt = (r[1] or "").strip()
                if not txt:
                    W = conn.execute("SELECT norm_text FROM work WHERE id=?", (r[4],)).fetchone()
                    if W and W[0]:
                        cs, ce = int(r[2]), int(r[3])
                        txt = W[0][cs:ce]
                if txt:
                    support_texts.append(txt[:480])

        # --- 4) Build trope definition block (annotated with prior weight) --
        defs = []
        qmarks = ",".join(["?"] * len(avail_ids))
        rows = conn.execute(
            f"SELECT id, name, summary FROM trope WHERE id IN ({qmarks})",
            tuple(avail_ids)
        ).fetchall()
        for tid, name, summary in rows:
            w = float(weights.get(tid, 1.0))
            defs.append(f"- {tid} :: {name} — {summary or ''}  (PRIOR={w:.2f})")

        # --- 5) Prompt ------------------------------------------------------
        prior_json = json.dumps({tid: round(float(weights.get(tid, 1.0)), 3) for tid in avail_ids})
        support_block = "\n\nSupport snippets (chosen via rerank):\n" + \
                        ("\n---\n".join(support_texts) if support_texts else "(none)")
        prompt = (
            f"SYSTEM: {JUDGE_SYSTEM}\n\n"
            f"SCENE [chars {s}-{e}] (absolute offsets into work.norm_text):\n{scene_text[:2400]}\n"
            f"{support_block}\n\n"
            f"CANDIDATE TROPES (id :: name — summary, annotated with PRIOR):\n" + ("\n".join(defs)) + "\n\n"
            f"AVAILABLE_TROPE_IDS (use only these in output):\n" + json.dumps(avail_ids) + "\n\n"
            f"PRIOR_WEIGHTS (hint; multiply your internal score by these priors):\n{prior_json}\n\n"
            + JUDGE_INSTRUCTIONS.replace("THRESHOLD", str(threshold))
            + " Also: Use only values from AVAILABLE_TROPE_IDS for 'trope_id'. Do not invent new ids or names.\n"
        )

        resp = call_reasoner(ollama_url, reasoner_model, prompt)
        items = extract_json(resp)
        try:
            print(f"[judge] scene={scene_id[:8]} cand={len(avail_ids)} support={len(support_texts)} items={len(items) if items else 0}")
        except Exception:
            pass

        # --- 6) Insert findings (apply prior) ------------------------------
        finding_cols = get_table_columns(conn, "trope_finding")
        has_level = "level" in finding_cols

        for it in items or []:
            try:
                tid = it["trope_id"]
                raw_conf = float(it.get("confidence", 0.0))
                w = float(weights.get(tid, 1.0))
                adj_conf = max(0.0, min(1.0, raw_conf * w))
                if adj_conf < threshold:
                    continue

                # clamp spans to valid range (absolute into work.norm_text)
                span = it.get("evidence_char_span") or [s, min(e, s + len(scene_text))]
                ev_s, ev_e = int(span[0]), int(span[1])
                N = len(full_text)
                ev_s = max(0, min(ev_s, N))
                ev_e = max(0, min(ev_e, N))
                if ev_e < ev_s:
                    ev_s, ev_e = ev_e, ev_s

                rationale = (it.get("rationale", "") or "").strip()
                if w != 1.0:
                    rationale = (rationale + f" [prior={w:.2f}, raw={raw_conf:.2f}, adj={adj_conf:.2f}]").strip()

                fid = str(uuid.uuid4())
                if has_level:
                    conn.execute(
                        "INSERT OR REPLACE INTO trope_finding("
                        " id, work_id, scene_id, chunk_id, trope_id, level, confidence,"
                        " evidence_start, evidence_end, rationale, model)"
                        " VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                        (fid, work_id, scene_id, None, tid, "scene", adj_conf, ev_s, ev_e, rationale, reasoner_model)
                    )
                else:
                    conn.execute(
                        "INSERT OR REPLACE INTO trope_finding("
                        " id, work_id, scene_id, chunk_id, trope_id, confidence,"
                        " evidence_start, evidence_end, rationale, model)"
                        " VALUES(?,?,?,?,?,?,?,?,?,?)",
                        (fid, work_id, scene_id, None, tid, adj_conf, ev_s, ev_e, rationale, reasoner_model)
                    )
                inserted += 1
            except Exception as e:
                print(f"[judge] scene={scene_id[:8]} skip item due to error: {e}; item={it}")
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

    p_j = sub.add_parser("judge-scenes", help="LLM-based scene judgments with retrieval + rerank + sanity")
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
    p_j.add_argument("--top-k", type=int, default=8)
    p_j.add_argument("--threshold", type=float, default=0.55)

    return p


def main():
    ap = build_cli()
    args = ap.parse_args()
    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row  # for dict-style access
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
