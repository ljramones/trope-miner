#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rerank & sanity layer for Trope Miner.

Pipeline:
  1) Retrieve top-K chunk candidates from Chroma.
     - If PER_WORK_COLLECTIONS=1: query collection "<CHUNK_COLLECTION>__<work_id>".
       If it has no results, fall back to the global collection with where={"work_id": ...}.
     - Else (default): query the global collection with where={"work_id": ...}.
  2) Ask a local LLM to pick the M most relevant snippets for judging.
     (We pass stage-1 KNN similarity and discourage generic background.)
  3) Persist selections:
       - scene_support        (summary: chosen ids + notes)
       - support_selection    (per-chosen-chunk: rank + stage1/2 scores)
  4) Compute trope "sanity" priors (lexical mention + semantic affinity)
     and persist them to trope_sanity (one row per (scene, trope)).
  5) Return (chosen_support_ids, weights_by_trope_id) to the judge.

All I/O stays local (SQLite + Chroma + Ollama).
"""
from __future__ import annotations

import os
import re
import json
import math
import sqlite3
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import requests
import chromadb  # pip install chromadb


# ----------------- env/config -----------------
def _truthy(v: Optional[str]) -> bool:
    return (v or "").strip().lower() in {"1", "true", "yes", "on"}

OLLAMA = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
REASONER_MODEL = os.getenv("REASONER_MODEL", "llama3.1:8b")   # or qwen2.5:7b-instruct
EMBED_MODEL    = os.getenv("EMBED_MODEL", "nomic-embed-text")

CHROMA_HOST    = os.getenv("CHROMA_HOST", "127.0.0.1")
CHROMA_PORT    = int(os.getenv("CHROMA_PORT", "8000"))
CHUNK_COLLECTION = os.getenv("CHUNK_COLLECTION", "trope-miner-v1-cos")
PER_WORK_COLLECTIONS = _truthy(os.getenv("PER_WORK_COLLECTIONS", "0"))
HNSW_SPACE = os.getenv("CHROMA_SPACE", "cosine")  # usually "cosine"

# retrieval / rerank knobs
TOP_K        = int(os.getenv("RERANK_TOP_K", "8"))            # first-stage K from Chroma
KEEP_M       = int(os.getenv("RERANK_KEEP_M", "3"))           # LLM keeps top M
DOC_CHAR_MAX = int(os.getenv("RERANK_DOC_CHAR_MAX", "480"))   # per-snippet budget in prompt

# shortlist sanity knobs
DOWNWEIGHT_NO_MENTION = float(os.getenv("DOWNWEIGHT_NO_MENTION", "0.55"))
SEM_SIM_THRESHOLD     = float(os.getenv("SEM_SIM_THRESHOLD", "0.36"))  # cosine ~ rule of thumb


# --- persistence helpers: support_selection & trope_sanity (safety net) ---

def _ensure_support_tables(conn: sqlite3.Connection) -> None:
    """Safety: create detail tables if schema migration hasn't run yet."""
    conn.executescript("""
    PRAGMA foreign_keys=ON;

    CREATE TABLE IF NOT EXISTS support_selection (
      scene_id     TEXT NOT NULL,
      chunk_id     TEXT NOT NULL,
      rank         INTEGER NOT NULL,
      stage1_score REAL,     -- KNN similarity (1 - distance)
      stage2_score REAL,     -- rank-based score from LLM rerank
      picked       INTEGER NOT NULL DEFAULT 1,
      created_at   TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
      PRIMARY KEY (scene_id, chunk_id)
    );

    CREATE TABLE IF NOT EXISTS trope_sanity (
      scene_id   TEXT NOT NULL,
      trope_id   TEXT NOT NULL,
      lex_ok     INTEGER NOT NULL,  -- 1 if name/alias mentioned (scene or support)
      sem_sim    REAL    NOT NULL,  -- semantic similarity 0..1
      weight     REAL    NOT NULL,  -- prior multiplied with model score
      created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
      PRIMARY KEY (scene_id, trope_id)
    );
    """)
    conn.commit()


def _persist_support_selection(
    conn: sqlite3.Connection,
    scene_id: str,
    ordered_chunk_ids: List[str],
    stage1_scores: Optional[Dict[str, float]] = None,
    stage2_scores: Optional[Dict[str, float]] = None,
) -> None:
    """
    Store (scene_id, chunk_id, rank, [stage1, stage2], picked=1) while preserving created_at.
    """
    _ensure_support_tables(conn)
    stage1_scores = stage1_scores or {}
    stage2_scores = stage2_scores or {}

    # Discover current table shape
    cols = {row[1] for row in conn.execute("PRAGMA table_info(support_selection)").fetchall()}
    use_stage1 = "stage1_score" in cols
    use_stage2 = "stage2_score" in cols

    # Build column list and upsert SQL (avoid REPLACE to preserve created_at)
    all_cols = ["scene_id", "chunk_id", "rank"]
    if use_stage1:
        all_cols.append("stage1_score")
    if use_stage2:
        all_cols.append("stage2_score")
    all_cols.append("picked")

    placeholders = ",".join("?" for _ in all_cols)
    sql = (
        f"INSERT INTO support_selection ({','.join(all_cols)}) VALUES ({placeholders}) "
        "ON CONFLICT(scene_id, chunk_id) DO UPDATE SET "
        "  rank=excluded.rank"
        + (", stage1_score=excluded.stage1_score" if use_stage1 else "")
        + (", stage2_score=excluded.stage2_score" if use_stage2 else "")
        + ", picked=excluded.picked"
    )

    rows = []
    for rnk, cid in enumerate(ordered_chunk_ids, start=1):
        vals = [scene_id, cid, rnk]
        if use_stage1:
            vals.append(stage1_scores.get(cid))
        if use_stage2:
            vals.append(stage2_scores.get(cid))
        vals.append(1)  # picked
        rows.append(tuple(vals))

    if rows:
        conn.executemany(sql, rows)
        conn.commit()


def _persist_trope_sanity(
    conn: sqlite3.Connection,
    scene_id: str,
    weights: Dict[str, float],
    lex_ok_by_trope: Optional[Dict[str, int]] = None,
    sem_sim_by_trope: Optional[Dict[str, float]] = None,
) -> None:
    """Store one row per (scene_id, trope_id) with lex_ok, sem_sim, weight."""
    _ensure_support_tables(conn)
    lex_ok_by_trope = lex_ok_by_trope or {}
    sem_sim_by_trope = sem_sim_by_trope or {}
    if not weights:
        return

    sql = (
        "INSERT INTO trope_sanity(scene_id, trope_id, lex_ok, sem_sim, weight) "
        "VALUES (?,?,?,?,?) "
        "ON CONFLICT(scene_id, trope_id) DO UPDATE SET "
        "  lex_ok=excluded.lex_ok, sem_sim=excluded.sem_sim, weight=excluded.weight"
    )
    rows = []
    for tid, w in weights.items():
        rows.append((
            scene_id,
            tid,
            int(lex_ok_by_trope.get(tid, 0)),
            float(sem_sim_by_trope.get(tid, 0.0)),
            float(w),
        ))
    conn.executemany(sql, rows)
    conn.commit()


# ----------------- small utils -----------------
def cosine(a: List[float], b: List[float]) -> float:
    num = sum(x * y for x, y in zip(a, b))
    da  = math.sqrt(sum(x * x for x in a)); db = math.sqrt(sum(y * y for y in b))
    return 0.0 if da == 0 or db == 0 else num / (da * db)


def ollama_embed(texts: List[str]) -> List[List[float]]:
    """Embed a list of texts with Ollama embeddings endpoint."""
    url = f"{OLLAMA.rstrip('/')}/api/embeddings"
    out: List[List[float]] = []
    for t in texts:
        try:
            # Accepts 'prompt'; some builds also accept 'input'
            r = requests.post(url, json={"model": EMBED_MODEL, "prompt": t}, timeout=90)
            r.raise_for_status()
        except Exception as e:
            raise RuntimeError(f"ollama_embed failed (model={EMBED_MODEL}, len={len(t)}): {e}")
        data = r.json()
        emb = data.get("embedding")
        if not emb and isinstance(data.get("data"), list) and data["data"]:
            emb = data["data"][0].get("embedding")
        out.append(emb or [])
    return out


def ollama_json(model: str, prompt: str, system: Optional[str] = None) -> dict:
    """Call Ollama generate; parse the first {...} block as JSON."""
    url = f"{OLLAMA.rstrip('/')}/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": False, "temperature": 0.2}
    if system:
        payload["system"] = system
    r = requests.post(url, json=payload, timeout=180)
    r.raise_for_status()
    txt = r.json().get("response", "") or ""
    # Prefer fenced ```json ... ```; else first minimal JSON object
    m = re.search(r"```json\s*(\{.*?\})\s*```", txt, flags=re.DOTALL)
    if not m:
        m = re.search(r"\{.*?\}", txt, flags=re.DOTALL)
    if not m:
        return {}
    try:
        blob = m.group(1) if m.lastindex == 1 else m.group(0)
        return json.loads(blob)
    except Exception:
        return {}


def safe_trunc(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).lower()


# ----------------- data types -----------------
@dataclass
class ChunkHit:
    id: str
    text: str
    dist: float   # Chroma distance (cosine distance ~ 1 - similarity)
    meta: dict


# ----------------- Chroma retrieval -----------------
class ChunkRetriever:
    def __init__(self):
        self.client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        # Global collection (for fallback / default)
        self.global_col = self.client.get_or_create_collection(
            CHUNK_COLLECTION, metadata={"hnsw:space": HNSW_SPACE}
        )

    def _per_work_collection(self, work_id: str):
        """Return the per-work collection (creating if absent)."""
        name = f"{CHUNK_COLLECTION}__{work_id}"
        return self.client.get_or_create_collection(name, metadata={"hnsw:space": HNSW_SPACE})

    def _query(self, col, q_emb, k: int, where: Optional[dict] = None) -> dict:
        include = ["documents", "distances", "metadatas"]
        if where:
            return col.query(query_embeddings=[q_emb], n_results=k, include=include, where=where)
        return col.query(query_embeddings=[q_emb], n_results=k, include=include)

    def topk_for_scene(self, work_id: str, scene_text: str, k: int = TOP_K) -> List[ChunkHit]:
        """Return top-k hits with text, distance, metadata (preferring per-work collection if enabled)."""
        [q_emb] = ollama_embed([scene_text])

        # Try per-work collection if toggled on
        res = None
        if PER_WORK_COLLECTIONS:
            try:
                pw_col = self._per_work_collection(work_id)
                res = self._query(pw_col, q_emb, k)
                ids_try = (res.get("ids") or [[]])[0] if isinstance(res.get("ids"), list) else []
                if not ids_try:  # empty -> fall back to global w/ where
                    res = None
            except Exception:
                res = None  # fall through to global

        if res is None:
            # Global collection with per-work filter
            res = self._query(self.global_col, q_emb, k, where={"work_id": work_id})

        ids   = (res.get("ids") or [[]])[0]
        docs  = (res.get("documents") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]

        hits: List[ChunkHit] = []
        for i, doc_id in enumerate(ids):
            meta = metas[i] if i < len(metas) else {}
            chunk_id = str(meta.get("chunk_id") or doc_id)
            text = docs[i] if i < len(docs) and docs[i] is not None else ""
            dist = float(dists[i]) if i < len(dists) and dists[i] is not None else 1.0
            hits.append(ChunkHit(id=chunk_id, text=text, dist=dist, meta=meta or {}))
        return hits


# ----------------- LLM rerank -----------------
def rerank_chunks_with_llm(
    scene_text: str, hits: List[ChunkHit], keep_m: int = KEEP_M
) -> Tuple[List[str], str, Dict[str, float]]:
    """
    Ask the LLM to select the M most helpful snippets.
    Returns (chosen_ids, rationale_text, stage2_scores_by_id).
    Includes stage-1 KNN similarity as side info to bias against weak/generic snippets.
    """
    items = []
    for h in hits:
        knn_sim = max(0.0, min(1.0, 1.0 - float(h.dist)))
        items.append({
            "id": h.id,
            "knn": round(knn_sim, 3),
            "len": len(h.text or ""),
            "snippet": safe_trunc((h.text or "").strip(), DOC_CHAR_MAX),
        })

    sys = (
        "You help pick the most directly relevant snippets to judge which narrative tropes are present in a scene. "
        "Prefer snippets with concrete, local evidence (actions, claims, dialogue) over generic background. "
        "If two snippets are equally relevant, prefer the one with the higher KNN score."
    )

    prompt = f"""Scene (trimmed):
\"\"\"{safe_trunc(scene_text, 2500)}\"\"\"

Candidate snippets:
Each item has: id, knn (KNN similarity from 0..1), len, snippet.
{json.dumps(items, ensure_ascii=False, indent=2)}

Task:
- Choose the {min(keep_m, len(items))} snippets that are MOST directly useful as evidence.
- De-prioritize generic background that doesn't bear on trope judgments, even if long.
- When ties, prefer higher 'knn'.
- Return STRICT JSON ONLY:

{{
  "support_ids": ["<id1>", "<id2>", "..."],
  "notes": "one short reason describing why these were chosen"
}}
"""

    data = ollama_json(REASONER_MODEL, prompt, system=sys)
    chosen = data.get("support_ids") or []
    notes  = (data.get("notes") or "").strip()

    if not chosen:  # fallback to KNN order
        chosen = [h.id for h in hits[:keep_m]]
        notes = notes or "fallback=knn"

    # Keep only known ids and trim to keep_m
    allowed = {h.id for h in hits}
    chosen = [c for c in chosen if c in allowed][:keep_m]

    # simple rank-based score in [0,1]
    stage2_scores: Dict[str, float] = {}
    if chosen:
        M = len(chosen)
        for rank, cid in enumerate(chosen, start=1):
            stage2_scores[cid] = (M - rank + 1) / float(M)

    return chosen, notes, stage2_scores


# ----------------- Trope shortlist sanity -----------------
def load_trope_catalog(conn: sqlite3.Connection) -> Dict[str, dict]:
    rows = conn.execute("SELECT id, name, summary, aliases FROM trope").fetchall()
    out: Dict[str, dict] = {}
    for (tid, name, summary, aliases_blob) in rows:
        aliases: List[str] = []
        if aliases_blob:
            try:
                a = json.loads(aliases_blob)
                if isinstance(a, list):
                    aliases = [x for x in a if isinstance(x, str)]
            except Exception:
                aliases = []
        out[tid] = {"name": name or "", "summary": summary or "", "aliases": aliases}
    return out


def has_lexical_mention(text: str, phrases: List[str]) -> bool:
    s = normalize(text)
    for p in phrases:
        p2 = normalize(p)
        if not p2:
            continue
        if " " in p2:           # phrase
            if p2 in s: return True
        else:                   # single token
            if re.search(rf"\b{re.escape(p2)}\b", s): return True
    return False


def compute_sanity_metrics(
    conn: sqlite3.Connection,
    scene_text: str,
    support_texts: List[str],
    candidate_trope_ids: List[str],
) -> Dict[str, dict]:
    """
    Returns per-trope metrics: {tid: {'lex_ok':0|1, 'sem_sim':float, 'weight':float}}
    Uses SAME thresholds/constants as sanity_downweights().
    """
    cat = load_trope_catalog(conn)
    support_joined = " ".join(support_texts)

    wanted = [tid for tid in candidate_trope_ids if tid in cat]
    texts_to_embed = [scene_text, support_joined] + [
        f"{cat[tid]['name']}. {cat[tid]['summary']}" for tid in wanted
    ]
    embs = ollama_embed(texts_to_embed)
    scene_emb = embs[0] if embs else []
    support_emb = embs[1] if len(embs) > 1 else []
    trope_vecs = {tid: embs[2 + i] for i, tid in enumerate(wanted)}

    metrics: Dict[str, dict] = {}
    for tid in candidate_trope_ids:
        t = cat.get(tid)
        if not t:
            continue
        phrases = [t["name"]] + (t["aliases"] or [])
        lex = has_lexical_mention(scene_text, phrases) or has_lexical_mention(support_joined, phrases)
        sem = 0.0
        if tid in trope_vecs and scene_emb and support_emb:
            sem = max(cosine(scene_emb, trope_vecs[tid]), cosine(support_emb, trope_vecs[tid]))
        w = 1.0 if (lex or sem >= SEM_SIM_THRESHOLD) else DOWNWEIGHT_NO_MENTION
        metrics[tid] = {"lex_ok": 1 if lex else 0, "sem_sim": float(sem), "weight": float(w)}
    return metrics


def sanity_downweights(
    conn: sqlite3.Connection,
    scene_text: str,
    support_texts: List[str],
    candidate_trope_ids: List[str],
) -> Dict[str, float]:
    """Compatibility shim: return only weights by calling compute_sanity_metrics()."""
    metrics = compute_sanity_metrics(conn, scene_text, support_texts, candidate_trope_ids)
    return {tid: m["weight"] for tid, m in metrics.items()}


# ----------------- scene_support (summary) -----------------
def ensure_scene_support_schema(conn: sqlite3.Connection) -> None:
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS scene_support(
      scene_id    TEXT NOT NULL,
      support_ids TEXT NOT NULL,  -- JSON array of chunk ids
      notes       TEXT,
      model       TEXT,
      k           INTEGER,
      m           INTEGER,
      created_at  TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
      PRIMARY KEY(scene_id)
    );
    """)
    conn.commit()


def persist_scene_support(conn: sqlite3.Connection, scene_id: str, support_ids: List[str], notes: str):
    ensure_scene_support_schema(conn)
    conn.execute("""
      INSERT INTO scene_support(scene_id, support_ids, notes, model, k, m)
      VALUES(?,?,?,?,?,?)
      ON CONFLICT(scene_id) DO UPDATE SET
        support_ids=excluded.support_ids, notes=excluded.notes,
        model=excluded.model, k=excluded.k, m=excluded.m,
        created_at=strftime('%Y-%m-%dT%H:%M:%fZ','now')
    """, (scene_id, json.dumps(support_ids), notes, REASONER_MODEL, TOP_K, KEEP_M))
    conn.commit()


# ----------------- Orchestrator -----------------
def choose_support_and_sanity(
    conn: sqlite3.Connection,
    work_id: str,
    scene_id: str,
    scene_text: str,
    candidate_trope_ids: List[str],
    persist: bool = True,
) -> Tuple[List[str], Dict[str, float]]:
    """
    Returns (support_chunk_ids, weights_by_trope_id).

    Side effects (if persist=True):
      - writes scene_support (summary of chosen support ids + notes)
      - writes support_selection rows (rank + stage1/2 scores if those columns exist)
      - writes trope_sanity (lex_ok, sem_sim, weight) for each candidate trope
    """
    # ---------------- Stage 1: retrieval ----------------
    retr = ChunkRetriever()
    hits = retr.topk_for_scene(work_id, scene_text, TOP_K)

    # Stage-1 scores (convert distance → similarity ∈ [0,1])
    stage1_scores: Dict[str, float] = {}
    for h in hits:
        sim = 1.0 - float(h.dist)
        stage1_scores[h.id] = max(0.0, min(1.0, sim))

    # ---------------- Stage 2: LLM rerank ------------------------
    ret = rerank_chunks_with_llm(scene_text, hits, KEEP_M)
    if isinstance(ret, tuple) and len(ret) == 3:
        chosen_ids, notes, stage2_scores = ret
    elif isinstance(ret, tuple) and len(ret) == 2:
        chosen_ids, notes = ret
        stage2_scores = {cid: None for cid in (chosen_ids or [])}
    else:  # ultra-defensive fallback
        chosen_ids, notes, stage2_scores = [h.id for h in hits[:KEEP_M]], "fallback=knn", {}

    # ---------------- Persistence: support choices ---------------
    if persist:
        persist_scene_support(conn, scene_id, chosen_ids, notes)
        _persist_support_selection(conn, scene_id, chosen_ids, stage1_scores, stage2_scores)

    # ---------------- Sanity metrics & priors --------------------
    id2text = {h.id: (h.text or "") for h in hits}
    chosen_texts = [id2text.get(cid, "") for cid in chosen_ids]

    metrics = compute_sanity_metrics(conn, scene_text, chosen_texts, candidate_trope_ids)
    if persist and metrics:
        weights = {tid: m["weight"] for tid, m in metrics.items()}
        lex_ok  = {tid: m["lex_ok"]  for tid, m in metrics.items()}
        sem_sim = {tid: m["sem_sim"] for tid, m in metrics.items()}
        _persist_trope_sanity(conn, scene_id, weights, lex_ok, sem_sim)

    # Return only the weights dict to the judge (default 1.0 if missing)
    weights_out = {tid: float(metrics.get(tid, {}).get("weight", 1.0)) for tid in candidate_trope_ids}
    return chosen_ids, weights_out
