#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Seed trope candidates via semantic similarity against chunk embeddings.

For each trope:
  - Build a query text: "<name>. <summary>" (fallback to first 2–3 aliases).
  - Embed with Ollama.
  - Query Chroma chunk collection with where={"work_id": WORK_ID}, n_results=TOP_N.
  - Keep results with similarity >= TAU (1 - distance).
  - Insert into trope_candidate with:
        id=lower(hex(randomblob(16))),
        work_id, scene_id, chunk_id, trope_id,
        surface=NULL, alias=NULL,
        start=chunk.char_start, end=chunk.char_end,
        source='semantic', score=<similarity>
    Cap per (trope_id, scene_id) to PER_SCENE_CAP to avoid explosion.

Idempotent across runs via UNIQUE(work_id, trope_id, start, end).
"""

from __future__ import annotations
import argparse
import json
import sqlite3
from typing import Dict, List, Tuple, Optional

import requests
import chromadb  # pip install chromadb

# ----------------- Embedding (Ollama) -----------------
def ollama_embed(ollama_url: str, model: str, text: str, timeout: int = 90) -> List[float]:
    url = ollama_url.rstrip("/") + "/api/embeddings"
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
        raise RuntimeError("empty embedding from Ollama")
    return emb

# ----------------- Chroma -----------------
def get_collection(host: str, port: int, name: str):
    client = chromadb.HttpClient(host=host, port=port)
    try:
        return client.get_collection(name)
    except Exception:
        # create with cosine as a safe default
        return client.create_collection(name, metadata={"hnsw:space": "cosine"})

# ----------------- DB helpers -----------------
def ensure_indexes(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.execute("CREATE INDEX IF NOT EXISTS idx_candidate_work  ON trope_candidate(work_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_candidate_trope ON trope_candidate(trope_id)")
    cur.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_candidate_span "
        "ON trope_candidate(work_id, trope_id, start, end)"
    )
    conn.commit()

def load_tropes(conn: sqlite3.Connection) -> List[sqlite3.Row]:
    conn.row_factory = sqlite3.Row
    q = "SELECT id, name, COALESCE(summary,''), COALESCE(aliases,'') FROM trope ORDER BY name COLLATE NOCASE"
    return conn.execute(q).fetchall()

def chunk_index(conn: sqlite3.Connection, work_id: str) -> Dict[str, Tuple[str, int, int]]:
    """
    Return {chunk_id: (scene_id, char_start, char_end)} for quick lookups.
    """
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id, scene_id, char_start, char_end FROM chunk WHERE work_id=?",
        (work_id,),
    ).fetchall()
    return {r["id"]: (r["scene_id"], int(r["char_start"] or 0), int(r["char_end"] or 0)) for r in rows}

def trope_query_text(name: str, summary: str, aliases_json: str) -> str:
    name = (name or "").strip()
    summary = (summary or "").strip()
    if summary:
        return f"{name}. {summary}"
    # fallback: first 2–3 aliases (if any)
    try:
        arr = json.loads(aliases_json) if aliases_json else []
        if isinstance(arr, list):
            al = [x for x in arr if isinstance(x, str)]
            if al:
                return f"{name}. " + "; ".join(al[:3])
    except Exception:
        pass
    return name

# ----------------- Main logic -----------------
def main():
    ap = argparse.ArgumentParser(description="Semantic candidate seeding via Chroma similarity.")
    ap.add_argument("--db", required=True)
    ap.add_argument("--work-id", required=True)
    ap.add_argument("--collection", required=True, help="Chroma collection name for CHUNKS")
    ap.add_argument("--chroma-host", default="localhost")
    ap.add_argument("--chroma-port", type=int, default=8000)
    ap.add_argument("--embed-model", default="nomic-embed-text")
    ap.add_argument("--ollama-url", default="http://localhost:11434")
    ap.add_argument("--top-n", type=int, default=8, help="Chroma top-N per trope")
    ap.add_argument("--tau", type=float, default=0.70, help="similarity threshold (1 - distance)")
    ap.add_argument("--per-scene-cap", type=int, default=3, help="max semantic seeds per (trope, scene)")
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    ensure_indexes(conn)

    # Preload chunk spans for fast inserts
    cidx = chunk_index(conn, args.work_id)
    if not cidx:
        raise SystemExit(f"No chunks found for work {args.work_id}; did you ingest?")

    col = get_collection(args.chroma_host, args.chroma_port, args.collection)
    tropes = load_tropes(conn)

    inserted = 0
    per_scene_counts: Dict[Tuple[str, str], int] = {}  # (trope_id, scene_id) -> count
    cur = conn.cursor()

    for tr in tropes:
        tid = tr["id"]
        qtext = trope_query_text(tr["name"], tr[2], tr[3])
        if not qtext:
            continue

        try:
            q_emb = ollama_embed(args.ollama_url, args.embed_model, qtext)
        except Exception as ex:
            print(f"[seed-sem] skip trope {tid[:8]} (embed fail): {ex}")
            continue

        try:
            res = col.query(
                query_embeddings=[q_emb],
                n_results=max(1, args.top_n),
                where={"work_id": args.work_id},
                include=["metadatas", "distances"],
            )
        except Exception as ex:
            print(f"[seed-sem] query failed for trope {tid[:8]}: {ex}")
            continue

        ids = (res.get("ids") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]

        # Normalize into (chunk_id, sim, scene_id, start, end)
        seen_chunks = set()
        candidates: List[Tuple[str, float, str, int, int]] = []
        for i, doc_id in enumerate(ids):
            meta = metas[i] if i < len(metas) else {}
            chunk_id = str(meta.get("chunk_id") or doc_id)
            if not chunk_id or chunk_id in seen_chunks:
                continue
            seen_chunks.add(chunk_id)
            dist = float(dists[i]) if i < len(dists) and dists[i] is not None else 1.0
            sim = max(0.0, min(1.0, 1.0 - dist))
            if sim < args.tau:
                continue
            # find scene/span
            scene_id, cs, ce = cidx.get(chunk_id, (meta.get("scene_id"), meta.get("char_start") or 0, meta.get("char_end") or 0))
            if not scene_id:
                # fallback: skip if we truly have no scene_id
                continue
            candidates.append((chunk_id, sim, scene_id, int(cs), int(ce)))

        # enforce per-scene cap for this trope
        if not candidates:
            continue
        # sort by sim descending
        candidates.sort(key=lambda x: x[1], reverse=True)

        for chunk_id, sim, scene_id, cs, ce in candidates:
            key = (tid, scene_id)
            cnt = per_scene_counts.get(key, 0)
            if cnt >= args.per_scene_cap:
                continue
            try:
                cur.execute(
                    "INSERT OR IGNORE INTO trope_candidate("
                    " id, work_id, scene_id, chunk_id, trope_id, surface, alias, start, end, source, score"
                    ") VALUES (lower(hex(randomblob(16))), ?, ?, ?, ?, NULL, NULL, ?, ?, 'semantic', ?)",
                    (args.work_id, scene_id, chunk_id, tid, cs, ce, float(sim)),
                )
                if cur.rowcount:
                    inserted += 1
                    per_scene_counts[key] = cnt + 1
            except sqlite3.IntegrityError:
                # Duplicate (work,trope,start,end) — ignore
                pass

    conn.commit()
    print(f"[seed-sem] inserted {inserted} semantic candidates for work {args.work_id} "
          f"(tau={args.tau}, top_n={args.top_n}, cap/scene={args.per_scene_cap})")

if __name__ == "__main__":
    main()
