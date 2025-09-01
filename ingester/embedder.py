#!/usr/bin/env python3
"""
Trope Miner — Embedder
----------------------

Reads un-embedded chunks from SQLite and upserts embeddings into a Chroma
collection using Ollama's embeddings API (e.g., nomic-embed-text).

Usage
  $ export OLLAMA_BASE_URL=http://localhost:11434
  # Global collection (default):
  $ python embedder.py \
      --db ./tropes.db \
      --collection trope-miner-v1-cos \
      --model nomic-embed-text \
      --chroma-host localhost --chroma-port 8000 \
      --batch-size 64 --limit 0

  # Per-work collections (one collection per work):
  $ PER_WORK_COLLECTIONS=1 python embedder.py ...    # or: --per-work-collections

Optionally verify with a query:
  $ python embedder.py --db ./tropes.db --collection trope-miner-v1-cos \
      --query "vengeful spirit stalking the night" --top-k 5

Notes
  • Canonical text lives in SQLite. Chroma stores vectors + documents + metadata.
  • Idempotent per collection: skips chunks already in embedding_ref.
  • Safe to re-run; only new chunks are embedded (per the target collection).
  • When PER_WORK_COLLECTIONS=1, each work is written to f"{collection}__{work_id}".
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


def get_unembedded_chunks(
    conn: sqlite3.Connection,
    collection: str,
    limit: int = 0,
    work_id: Optional[str] = None,
) -> List[ChunkRow]:
    """
    Return chunks that are NOT stamped in embedding_ref for the given collection.
    If work_id is provided, restrict to that work.
    """
    sql = (
        "SELECT c.id, c.work_id, c.scene_id, c.idx, c.char_start, c.char_end, c.text "
        "FROM chunk c LEFT JOIN embedding_ref e "
        "  ON e.chunk_id = c.id AND e.collection = ? "
        "WHERE e.chunk_id IS NULL "
    )
    params: List[object] = [collection]
    if work_id:
        sql += "AND c.work_id = ? "
        params.append(work_id)
    sql += "ORDER BY c.rowid ASC "
    if limit and limit > 0:
        sql += f"LIMIT {int(limit)}"

    rows = conn.execute(sql, tuple(params)).fetchall()
    return [ChunkRow(*r) for r in rows]


def list_work_ids(conn: sqlite3.Connection) -> List[str]:
    rows = conn.execute("SELECT DISTINCT work_id FROM chunk ORDER BY work_id").fetchall()
    return [r[0] for r in rows if r[0]]

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

def _flush_batch(
    conn: sqlite3.Connection,
    coll,
    collection: str,
    model: str,
    ids: List[str],
    docs: List[str],
    embs: List[List[float]],
    metas: List[Dict],
):
    """Upsert one prepared batch to Chroma and stamp embedding_ref."""
    if not ids:
        return

    # Defensive length check
    if not (len(ids) == len(embs) == len(metas) == len(docs)):
        print(f"!! Skipping batch due to length mismatch: ids={len(ids)} embs={len(embs)} metas={len(metas)} docs={len(docs)}")
        ids.clear(); embs.clear(); metas.clear(); docs.clear()
        return

    # Drop empties
    quads = [(i, d, e, m) for i, d, e, m in zip(ids, docs, embs, metas) if isinstance(e, list) and len(e) > 0]
    if not quads:
        print("!! Skipping batch: empty embeddings")
        ids.clear(); embs.clear(); metas.clear(); docs.clear()
        return

    ids_f, docs_f, embs_f, metas_f = zip(*quads)
    dim = len(embs_f[0]) if embs_f else 0
    if dim <= 0:
        print("!! Skipping batch: zero-length embedding")
        ids.clear(); embs.clear(); metas.clear(); docs.clear()
        return

    # Keep only vectors with consistent dimensionality
    keep = [(i, d, e, m) for (i, d, e, m) in zip(ids_f, docs_f, embs_f, metas_f) if len(e) == dim]
    if not keep:
        print("!! Skipping batch: inconsistent dims")
        ids.clear(); embs.clear(); metas.clear(); docs.clear()
        return
    ids_f, docs_f, embs_f, metas_f = zip(*keep)

    # Upsert to Chroma
    try:
        coll.upsert(
            ids=list(ids_f),
            embeddings=list(embs_f),
            documents=list(docs_f),
            metadatas=list(metas_f),
        )
    except Exception as e:
        print(f"!! Chroma upsert failed for batch of {len(ids_f)}: {e}")
        ids.clear(); embs.clear(); metas.clear(); docs.clear()
        return

    # Stamp embedding_ref
    stamped = [(cid, collection, model, dim, cid) for cid in ids_f]
    mark_embedded(conn, stamped)

    ids.clear(); embs.clear(); metas.clear(); docs.clear()


def embed_and_upsert(
    db_path: str,
    chroma_host: str,
    chroma_port: int,
    collection: str,
    ollama_url: str,
    model: str,
    batch_size: int,
    limit: int,
    space: str,
    per_work_collections: bool,
) -> None:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    client = chromadb.HttpClient(host=chroma_host, port=chroma_port)

    if per_work_collections:
        work_ids = list_work_ids(conn)
        if not work_ids:
            print("No chunks to embed.")
            return
        print(f"Embedding (per-work) using model '{model}' across {len(work_ids)} work(s).")
        grand_total = 0
        for w_id in work_ids:
            coll_name = f"{collection}__{w_id}"
            coll = get_chroma_collection(chroma_host, chroma_port, coll_name, space=space)
            rows = get_unembedded_chunks(conn, collection=coll_name, limit=limit, work_id=w_id)
            if not rows:
                print(f" .. work={w_id} up-to-date (0 to embed)")
                continue
            print(f" .. work={w_id} -> collection '{coll_name}'  (n={len(rows)})")

            ids: List[str] = []
            docs: List[str] = []
            embs: List[List[float]] = []
            metas: List[Dict] = []

            for i, ch in enumerate(rows, 1):
                try:
                    emb = embed_text_ollama(ollama_url, model, ch.text)
                except Exception as e:
                    print(f"!! Embedding failed for chunk {ch.id[:8]}…: {e}")
                    continue

                ids.append(ch.id)
                docs.append(ch.text if isinstance(ch.text, str) else "")
                embs.append(emb)
                metas.append({
                    "chunk_id": ch.id,
                    "work_id": ch.work_id,
                    "scene_id": ch.scene_id,
                    "chunk_idx": ch.idx,
                    "char_start": ch.char_start,
                    "char_end": ch.char_end,
                    "model": model,
                    "collection": coll_name,
                })

                if len(ids) >= batch_size:
                    _flush_batch(conn, coll, coll_name, model, ids, docs, embs, metas)
                    print(f" .. upserted {i}/{len(rows)}")

            # final flush per work
            _flush_batch(conn, coll, coll_name, model, ids, docs, embs, metas)
            print(f" .. done work={w_id} ({len(rows)} new embeddings)")
            grand_total += len(rows)

        print(f"Done. Embedded {grand_total} chunks across {len(work_ids)} per-work collection(s).")
        conn.close()
        return

    # -------- Global collection path (original behavior) --------
    rows = get_unembedded_chunks(conn, collection=collection, limit=limit)
    if not rows:
        print("Nothing to embed. All chunks are up-to-date.")
        conn.close()
        return

    print(f"Embedding {len(rows)} chunks → collection '{collection}' using model '{model}'")
    coll = get_chroma_collection(chroma_host, chroma_port, collection, space=space)

    start_t = time.time()
    ids: List[str] = []
    docs: List[str] = []
    embs: List[List[float]] = []
    metas: List[Dict] = []

    total = len(rows)
    for i, ch in enumerate(rows, 1):
        try:
            emb = embed_text_ollama(ollama_url, model, ch.text)
        except Exception as e:
            print(f"!! Embedding failed for chunk {ch.id[:8]}…: {e}")
            continue

        ids.append(ch.id)
        docs.append(ch.text if isinstance(ch.text, str) else "")
        embs.append(emb)
        metas.append({
            "chunk_id": ch.id,        # <— helps retrieval map back to SQLite
            "work_id": ch.work_id,
            "scene_id": ch.scene_id,
            "chunk_idx": ch.idx,
            "char_start": ch.char_start,
            "char_end": ch.char_end,
            "model": model,
            "collection": collection,
        })

        if len(ids) >= batch_size:
            _flush_batch(conn, coll, collection, model, ids, docs, embs, metas)
            print(f".. upserted {min(i, total)}/{total}")

    # final flush
    _flush_batch(conn, coll, collection, model, ids, docs, embs, metas)

    dur = time.time() - start_t
    print(f"Done. Embedded {total} chunks in {dur:.1f}s → collection '{collection}'.")
    conn.close()

# ----------------------------- Query helper (optional) ------------------

def query_demo(chroma_host: str, chroma_port: int, collection: str,
               ollama_url: str, model: str, text: str, top_k: int, space: str):
    coll = get_chroma_collection(chroma_host, chroma_port, collection, space=space)
    q_emb = embed_text_ollama(ollama_url, model, text)
    res = coll.query(query_embeddings=[q_emb], n_results=top_k, include=["metadatas", "documents", "distances"])
    ids = res.get("ids", [[]])[0]
    docs = res.get("documents", [[]])[0]
    dists = res.get("distances", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    for rank, (cid, dist, meta, doc) in enumerate(zip(ids, dists, metas, docs), 1):
        doc_preview = (doc or "")[:120].replace("\n", " ")
        print(f"#{rank} id={cid} dist={dist:.4f} meta={meta} doc≈{doc_preview!r}")

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
    p.add_argument("--limit", type=int, default=0, help="0 = all unembedded (per target collection)")
    p.add_argument("--query", help="Optional test query text")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--space", default="cosine", choices=["cosine", "l2", "ip"],
                   help="Vector distance space for the HNSW index (default: cosine)")
    p.add_argument("--per-work-collections", action="store_true",
                   default=(os.getenv("PER_WORK_COLLECTIONS", "0").lower() in {"1","true","yes"}),
                   help="Write embeddings into per-work collections named '<collection>__<work_id>'")
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
            per_work_collections=args.per_work_collections,
        )

if __name__ == "__main__":
    main()
