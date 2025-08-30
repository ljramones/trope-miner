#!/usr/bin/env python3
import argparse, sys
import chromadb

def _detect_dim(col):
    """Infer stored embedding dimension by fetching one vector (no server embedding)."""
    try:
        res = col.get(limit=1, include=["embeddings"])
        embs = (res or {}).get("embeddings") or []
        if embs and embs[0] is not None:
            return len(embs[0])
    except Exception:
        pass
    return None

def _get_one_vector(col):
    """Fetch (id, embedding) of one stored item for self-retrieval probe."""
    try:
        got = col.get(limit=1, include=["embeddings"])
        ids = (got or {}).get("ids") or []
        embs = (got or {}).get("embeddings") or []
        if ids and embs and embs[0] is not None:
            return ids[0], embs[0]
    except Exception:
        pass
    return None, None

def main():
    ap = argparse.ArgumentParser(description="Chroma sanity: count/space/dim per collection")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--collections", nargs="+", required=True)
    ap.add_argument("--expect-space", default="cosine",
                    help="Expected HNSW space (default: cosine)")
    ap.add_argument("--probe", action="store_true",
                    help="Run a self-retrieval probe using a stored vector (no server embedding).")
    ap.add_argument("--self-threshold", type=float, default=1e-3,
                    help="Max acceptable distance for self-retrieval (default: 1e-3)")
    args = ap.parse_args()

    client = chromadb.HttpClient(host=args.host, port=args.port)
    try:
        print("[INFO] server heartbeat:", client.heartbeat())
    except Exception:
        # Not fatal; keep going.
        pass

    ok = True
    dims_seen = set()

    for name in args.collections:
        try:
            col = client.get_collection(name)
        except Exception as e:
            print(f"[MISS] collection not found: {name} ({e})")
            ok = False
            continue

        # metadata / space
        try:
            md = col.metadata or {}
        except Exception:
            md = {}
        space = md.get("hnsw:space") or md.get("space") or "unknown"

        # count
        try:
            count = col.count()
        except Exception:
            count = "?"

        # stored dimension (only if we have at least one embedding stored)
        dim = None
        if isinstance(count, int) and count > 0:
            dim = _detect_dim(col)
            if dim is not None:
                dims_seen.add(dim)

        print(f"[OK] {name}  count={count}  space={space}  dim={dim if dim is not None else '?'}")
        if space != args.expect_space:
            print(f"     !! Expected space={args.expect_space} but got {space}. "
                  f"Set collection metadata 'hnsw:space' at creation time.")
            ok = False
        if space == "unknown":
            print("     ?? Collection metadata missing 'hnsw:space'.")
        if count == 0:
            print("     ?? Collection is empty; no stored embeddings to infer dimension.")

        # Optional: self-retrieval probe (no server-side embedding)
        if args.probe and isinstance(count, int) and count > 0:
            item_id, emb = _get_one_vector(col)
            if item_id is None or emb is None:
                print("     ?? Probe skipped: could not fetch a stored embedding (permissions or empty?).")
            else:
                try:
                    res = col.query(
                        query_embeddings=[emb],
                        n_results=1,
                        include=["ids", "distances"],
                    )
                    rid = (res.get("ids") or [[None]])[0][0]
                    d   = (res.get("distances") or [[None]])[0][0]
                    if rid != item_id:
                        print(f"     !! Self-probe returned id={rid!r}, expected {item_id!r}")
                        ok = False
                    if d is not None and d > args.self_threshold:
                        print(f"     ?? Self-probe distance is {d:.6f} (expected ~0.0). "
                              f"Index/space mismatch or vector normalization issue?")
                except Exception as e:
                    print(f"     probe failed: {e}")

    if len(dims_seen) > 1:
        ds = ", ".join(str(d) for d in sorted(dims_seen))
        print(f"[WARN] Multiple embedding dimensions detected across collections: {ds}. "
              f"Ensure your embedder matches each collection.")

    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
