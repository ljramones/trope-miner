#!/usr/bin/env python3
"""
Trope Miner — Embedder
----------------------

Reads un-embedded chunks from SQLite and upserts embeddings into a Chroma
collection using Ollama's embeddings API (e.g., nomic-embed-text).

Usage
  $ export OLLAMA_BASE_URL=http://localhost:11434
  $ python embedder.py \
      --db ./tropes.db \
      --collection trope-miner-v1 \
      --model nomic-embed-text \
      --chroma-host localhost --chroma-port 8000 \
      --batch-size 64 --limit 0

Optionally verify with a query:
  $ python embedder.py --db ./tropes.db --collection trope-miner-v1 \
      --query "vengeful spirit stalking the night" --top-k 5

Notes
  • Canonical text lives in SQLite. Chroma stores vectors + metadata.
  • Idempotent per collection: skips chunks already in embedding_ref.
  • Safe to re-run; only new chunks are embedded.
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import requests

# Chroma client (HTTP)
try:
    import chromadb
except Exception as e:  # pragma: no cover
    raise SystemExit("chromadb is required: pip install chromadb requests")

# ----------------------------- Ollama embeddings ------------------------

def embed_text_ollama(base_url: str, model: str, text: str, timeout: int = 120) -> List[float]:
    """Return a single embedding vector for the given text via Ollama.

    Tries both 'input' and 'prompt' keys for compatibility across Ollama versions.
    Raises on empty vectors.
    """
    url = base_url.rstrip("/") + "/api/embeddings"

    def _post(payload):
        r = requests.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        emb = None
        if isinstance(data, dict):
            emb = data.get("embedding")
            if emb is None and isinstance(data.get("data"), list) and data["data"]:
                emb = data["data"][0].get("embedding")
            if emb is None and isinstance(data.get("embeddings"), list) and data["embeddings"]:
                first = data["embeddings"][0]
                if isinstance(first, list):
                    emb = first
        return emb or []

    emb = _post({"model": model, "input": text})
    if not emb:
        emb = _post({"model": model, "prompt": text})  # compatibility path
    if not emb or not isinstance(emb, list):
        raise RuntimeError(f"Empty/invalid embedding from Ollama for model={model}")
    return emb

# ----------------------------- SQLite access ----------------------------

@dataclass
class ChunkRow:
    id: str
    work_id: str
    scene_id: Optional[str]
    idx: int
    char_start: int
    char_end: int
    text: str


def get_unembedded_chunks(conn: sqlite3.Connection, collection: str, limit: int = 0) -> List[ChunkRow]:
    sql = (
        "SELECT c.id, c.work_id, c.scene_id, c.idx, c.char_start, c.char_end, c.text "
        "FROM chunk c LEFT JOIN embedding_ref e "
        "  ON e.chunk_id = c.id AND e.collection = ? "
        "WHERE e.chunk_id IS NULL "
        "ORDER BY c.rowid ASC "
    )
    if limit and limit > 0:
        sql += f"LIMIT {int(limit)}"
    rows = conn.execute(sql, (collection,)).fetchall()
    return [ChunkRow(*r) for r in rows]


def mark_embedded(conn: sqlite3.Connection, rows: List[Tuple[str, str, str, int, str]]) -> None:
    """rows: (chunk_id, collection, model, dim, chroma_id)"""
    conn.executemany(
        "INSERT OR REPLACE INTO embedding_ref(chunk_id, collection, model, dim, chroma_id) "
        "VALUES(?,?,?,?,?)",
        rows,
    )
    conn.commit()

# ----------------------------- Chroma utils -----------------------------

def get_chroma_collection(host: str, port: int, name: str, space: str = "cosine"):
    """
    Get or create a Chroma collection. When creating, set the HNSW distance
    space explicitly via metadata={"hnsw:space": space}.
    """
    client = chromadb.HttpClient(host=host, port=port)
    try:
        col = client.get_collection(name)
        md = (col.metadata or {})
        got_space = md.get("hnsw:space") or md.get("space")
        if got_space and got_space != space:
            print(f"!! WARNING: collection '{name}' has hnsw:space={got_space}, expected {space}")
        return col
    except Exception:
        return client.create_collection(name, metadata={"hnsw:space": space})



# ----------------------------- Main embed loop --------------------------

def embed_and_upsert(
    db_path: str,
    chroma_host: str,
    chroma_port: int,
    collection: str,
    ollama_url: str,
    model: str,
    batch_size: int,
    limit: int,
    space: str,   # <— added
) -> None:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    chunks = get_unembedded_chunks(conn, collection=collection, limit=limit)
    if not chunks:
        print("Nothing to embed. All chunks are up-to-date.")
        return
    print(f"Embedding {len(chunks)} chunks → collection '{collection}' using model '{model}'")
    coll = get_chroma_collection(chroma_host, chroma_port, collection, space=space)

    start_t = time.time()
    embedded_rows: List[Tuple[str, str, str, int, str]] = []

    ids: List[str] = []
    embs: List[List[float]] = []
    metas: List[Dict] = []

    def flush():
        nonlocal ids, embs, metas, embedded_rows
        if not ids:
            return

        # Defensive: keep triples aligned
        if len(ids) != len(embs) or len(ids) != len(metas):
            print(f"!! Skipping batch due to length mismatch: ids={len(ids)} embs={len(embs)} metas={len(metas)}")
            ids.clear();
            embs.clear();
            metas.clear();
            return

        # Drop empties
        triples = [(i, e, m) for i, e, m in zip(ids, embs, metas) if isinstance(e, list) and len(e) > 0]
        if not triples:
            print("!! Skipping batch: empty embeddings")
            ids.clear();
            embs.clear();
            metas.clear();
            return

        ids_f, embs_f, metas_f = zip(*triples)

        # Ensure consistent dimensionality
        dim = len(embs_f[0])
        if dim <= 0:
            print("!! Skipping batch: zero-length embedding")
            ids.clear();
            embs.clear();
            metas.clear();
            return

        keep = [(i, e, m) for (i, e, m) in zip(ids_f, embs_f, metas_f) if len(e) == dim]
        if not keep:
            print("!! Skipping batch: inconsistent dims")
            ids.clear();
            embs.clear();
            metas.clear();
            return
        ids_f, embs_f, metas_f = zip(*keep)

        # Upsert to Chroma
        try:
            coll.upsert(ids=list(ids_f), embeddings=list(embs_f), metadatas=list(metas_f))
        except Exception as e:
            print(f"!! Chroma upsert failed for batch of {len(ids_f)}: {e}")
            ids.clear();
            embs.clear();
            metas.clear()
            return

        # Stamp embedding_ref (mark success) — use the outer 'model' value
        stamped = [(cid, collection, model, dim, cid) for cid in ids_f]
        mark_embedded(conn, stamped)
        embedded_rows.extend(stamped)

        ids.clear();
        embs.clear();
        metas.clear()

    total = len(chunks)
    for i, ch in enumerate(chunks, 1):
        try:
            emb = embed_text_ollama(ollama_url, model, ch.text)
        except Exception as e:
            print(f"!! Embedding failed for chunk {ch.id[:8]}…: {e}")
            continue

        ids.append(ch.id)
        embs.append(emb)
        metas.append({
            "work_id": ch.work_id,
            "scene_id": ch.scene_id,
            "chunk_idx": ch.idx,
            "char_start": ch.char_start,
            "char_end": ch.char_end,
            "model": model,
            "collection": collection,
        })

        if len(ids) >= batch_size:
            flush()
            print(f".. upserted {min(i, total)}/{total}")

    # final flush
    flush()

    dur = time.time() - start_t
    print(f"Done. Embedded {len(embedded_rows)} chunks in {dur:.1f}s → collection '{collection}'.")

# ----------------------------- Query helper (optional) ------------------

def query_demo(chroma_host: str, chroma_port: int, collection: str,
               ollama_url: str, model: str, text: str, top_k: int, space: str):
    coll = get_chroma_collection(chroma_host, chroma_port, collection, space=space)
    q_emb = embed_text_ollama(ollama_url, model, text)
    res = coll.query(query_embeddings=[q_emb], n_results=top_k, include=["metadatas", "distances"])
    ids = res.get("ids", [[]])[0]
    dists = res.get("distances", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    for rank, (cid, dist, meta) in enumerate(zip(ids, dists, metas), 1):
        print(f"#{rank} id={cid} dist={dist:.4f} meta={meta}")

# ----------------------------- CLI -------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Embed chunks into Chroma via Ollama embeddings")
    p.add_argument("--db", required=True)
    p.add_argument("--collection", required=True)
    p.add_argument("--model", default="nomic-embed-text")
    p.add_argument("--chroma-host", default=os.getenv("CHROMA_HOST", "localhost"))
    p.add_argument("--chroma-port", type=int, default=int(os.getenv("CHROMA_PORT", "8000")))
    p.add_argument("--ollama-url", default=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--limit", type=int, default=0, help="0 = all unembedded")
    p.add_argument("--query", help="Optional test query text")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--space", default="cosine", choices=["cosine", "l2", "ip"],
                   help="Vector distance space for the HNSW index (default: cosine)")
    return p


def main():
    args = build_arg_parser().parse_args()
    if args.query:
        query_demo(args.chroma_host, args.chroma_port, args.collection,
                   args.ollama_url, args.model, args.query, args.top_k, args.space)
    else:
        embed_and_upsert(
            db_path=args.db,
            chroma_host=args.chroma_host,
            chroma_port=args.chroma_port,
            collection=args.collection,
            ollama_url=args.ollama_url,
            model=args.model,
            batch_size=args.batch_size,
            limit=args.limit,
            space=args.space,
        )

if __name__ == "__main__":
    main()
