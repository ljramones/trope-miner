#!/usr/bin/env python3
"""
Embed trope catalog into a Chroma collection using Ollama embeddings.

Defaults:
  - DB: ./tropes.db
  - Collection: trope-catalog-nomic-cos (cosine space)
  - Model: nomic-embed-text
  - Ollama: OLLAMA_BASE_URL or http://localhost:11434
  - Chroma: CHROMA_HOST/PORT env or localhost:8000

Examples:
  $ python embed_tropes.py --db ./tropes.db --collection trope-catalog-nomic-cos

  # Recreate collection from scratch:
  $ python embed_tropes.py --recreate

  # Limit & faster feedback:
  $ python embed_tropes.py --limit 50 --batch-size 32
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import time
from typing import Dict, List, Optional, Tuple

import requests

try:
    import chromadb
except Exception:
    raise SystemExit("chromadb is required: pip install chromadb requests")

# ----------------------------- Ollama embeddings ------------------------

def embed_text_ollama(base_url: str, model: str, text: str, timeout: int = 90) -> List[float]:
    """Get a single embedding vector from Ollama. Tries 'input' then 'prompt'."""
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

# ----------------------------- Chroma helpers ---------------------------

def get_or_create_collection(client, name: str, space: str = "cosine", recreate: bool = False):
    if recreate:
        try:
            client.delete_collection(name)
            print(f"[info] deleted existing collection: {name}")
        except Exception:
            pass
    try:
        col = client.get_collection(name)
        md = col.metadata or {}
        got_space = md.get("hnsw:space") or md.get("space") or "unknown"
        if got_space != space:
            print(f"[WARN] collection '{name}' uses space={got_space}, expected {space}")
        return col
    except Exception:
        pass
    return client.create_collection(name, metadata={"hnsw:space": space})

# ----------------------------- DB access --------------------------------

def fetch_tropes(conn: sqlite3.Connection, limit: int = 0) -> List[Tuple[str, str, str, Optional[str]]]:
    sql = "SELECT id, name, COALESCE(summary,''), aliases FROM trope ORDER BY name COLLATE NOCASE"
    if limit and limit > 0:
        sql += f" LIMIT {int(limit)}"
    return conn.execute(sql).fetchall()

# ----------------------------- Text & metadata --------------------------

def trope_text_and_meta(
    name: str,
    summary: str,
    aliases_json: Optional[str],
    include_summary: bool,
    include_aliases: bool,
    aliases_topn: int,
) -> Tuple[str, Dict]:
    """Build the text to embed and a **scalar-only** metadata dict."""
    aliases_list: List[str] = []
    if include_aliases and aliases_json:
        try:
            arr = json.loads(aliases_json)
            if isinstance(arr, list):
                aliases_list = [a for a in arr if isinstance(a, str)]
        except Exception:
            pass

    txt_parts = [name]
    if include_summary and summary:
        txt_parts.append("— " + summary.strip())
    if include_aliases and aliases_list:
        # include a small slice in the *text* to improve recall
        sl = [a.strip() for a in aliases_list if a and len(a) <= 40]
        if sl:
            txt_parts.append("Aliases: " + ", ".join(sl[:aliases_topn]))

    # --- IMPORTANT: only scalar metadata values for Chroma ---
    meta: Dict = {
        "name": name or "",
        "summary": summary or "",
        "alias_count": len(aliases_list),
        "aliases_json": json.dumps(aliases_list, ensure_ascii=True) if aliases_list else None,
        # optional: stamp the model later in main()
    }

    return "  ".join(txt_parts).strip(), meta

# ----------------------------- Flush ------------------------------------

def flush_batch(coll, ids: List[str], embs: List[List[float]], metas: List[Dict]) -> int:
    if not ids:
        return 0
    if len(ids) != len(embs) or len(ids) != len(metas):
        print(f"[skip] batch length mismatch ids={len(ids)} embs={len(embs)} metas={len(metas)}")
        ids.clear(); embs.clear(); metas.clear()
        return 0
    dim = len(embs[0])
    keep = [(i, e, m) for i, e, m in zip(ids, embs, metas) if isinstance(e, list) and len(e) == dim]
    if not keep:
        print("[skip] empty/inconsistent batch")
        ids.clear(); embs.clear(); metas.clear()
        return 0
    ids_f, embs_f, metas_f = zip(*keep)

    # ensure metas are scalar-only
    safe_metas = []
    for m in metas_f:
        safe = {}
        for k, v in m.items():
            if isinstance(v, (str, int, float, bool)) or v is None:
                safe[k] = v
            else:
                # final guard: coerce anything unexpected to string
                safe[k] = json.dumps(v, ensure_ascii=True)
        safe_metas.append(safe)

    try:
        coll.upsert(ids=list(ids_f), embeddings=list(embs_f), metadatas=safe_metas)
    except Exception as e:
        print(f"[error] Chroma upsert failed for batch of {len(ids_f)}: {e}")
        ids.clear(); embs.clear(); metas.clear()
        return 0
    n = len(ids_f)
    ids.clear(); embs.clear(); metas.clear()
    return n

# ----------------------------- CLI --------------------------------------

def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Embed trope catalog into a Chroma collection via Ollama embeddings")
    p.add_argument("--db", default="tropes.db")
    p.add_argument("--collection", default="trope-catalog-nomic-cos")
    p.add_argument("--model", default="nomic-embed-text")
    p.add_argument("--ollama-url", default=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    p.add_argument("--chroma-host", default=os.getenv("CHROMA_HOST", "localhost"))
    p.add_argument("--chroma-port", type=int, default=int(os.getenv("CHROMA_PORT", "8000")))
    p.add_argument("--space", default=os.getenv("CHROMA_SPACE", "cosine"),
                   help="HNSW distance metric (cosine|l2|ip). Default: cosine")
    p.add_argument("--timeout", type=int, default=int(os.getenv("EMBED_TIMEOUT", "90")),
                   help="Ollama HTTP timeout (seconds)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--limit", type=int, default=0, help="Limit number of tropes (0=all)")
    p.add_argument("--recreate", action="store_true", help="Delete and recreate the collection")
    p.add_argument("--no-summary", action="store_true", help="Do not include summary text")
    p.add_argument("--no-aliases", action="store_true", help="Do not include aliases in the text")
    p.add_argument("--aliases-topn", type=int, default=12, help="Max aliases to include in text")
    return p.parse_args()

# ----------------------------- Main -------------------------------------

def main():
    args = build_args()

    client = chromadb.HttpClient(host=args.chroma_host, port=args.chroma_port)
    coll = get_or_create_collection(client, args.collection, space=args.space, recreate=args.recreate)

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    rows = fetch_tropes(conn, limit=args.limit)
    total = len(rows)
    if not rows:
        print("[info] no tropes found")
        return

    t0 = time.time()
    ids: List[str] = []
    embs: List[List[float]] = []
    metas: List[Dict] = []
    done = 0

    include_summary = not args.no_summary
    include_aliases = not args.no_aliases

    print(f"Embedding {total} trope definitions → '{args.collection}' (model={args.model})")

    for i, (tid, name, summary, aliases) in enumerate(rows, 1):
        text, meta = trope_text_and_meta(
            name=name,
            summary=summary or "",
            aliases_json=aliases,
            include_summary=include_summary,
            include_aliases=include_aliases,
            aliases_topn=args.aliases_topn,
        )
        # stamp model into (scalar) metadata
        meta["model"] = args.model

        try:
            vec = embed_text_ollama(args.ollama_url, args.model, text, timeout=args.timeout)
        except Exception as e:
            print(f"[warn] embedding failed for {name!r} ({tid[:8]}…): {e}")
            continue

        ids.append(str(tid))
        embs.append(vec)
        metas.append(meta)

        if len(ids) >= args.batch_size:
            done += flush_batch(coll, ids, embs, metas)
            print(f".. upserted {min(i, total)}/{total}")

    # final flush
    done += flush_batch(coll, ids, embs, metas)

    dt = time.time() - t0
    print(f"[OK] Upserted {done}/{total} trope defs into {args.collection} in {dt:.1f}s")

if __name__ == "__main__":
    main()
