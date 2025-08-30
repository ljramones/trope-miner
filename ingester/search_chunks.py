#!/usr/bin/env python3
import argparse, os, sqlite3, chromadb, requests

def embed(ollama_url: str, model: str, text: str):
    url = ollama_url.rstrip("/") + "/api/embeddings"
    r = requests.post(url, json={"model": model, "input": text}, timeout=60)
    data = r.json()
    emb = data.get("embedding") or (data.get("data") or [{}])[0].get("embedding")
    if not emb:
        r = requests.post(url, json={"model": model, "prompt": text}, timeout=60)
        data = r.json()
        emb = data.get("embedding") or (data.get("data") or [{}])[0].get("embedding")
    if not emb: raise RuntimeError(f"no embedding from {model}")
    return emb

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--collection", required=True)
    ap.add_argument("--model", default="nomic-embed-text")
    ap.add_argument("--ollama-url", default=os.getenv("OLLAMA_BASE_URL","http://localhost:11434"))
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("query")
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    client = chromadb.HttpClient(host=args.host, port=args.port)
    coll = client.get_collection(args.collection)

    q = embed(args.ollama_url, args.model, args.query)
    res = coll.query(query_embeddings=[q], n_results=args.top_k, include=["metadatas","distances"])
    ids = res.get("ids", [[]])[0]; dists = res.get("distances", [[]])[0]; metas = res.get("metadatas", [[]])[0]
    for rank, (cid, dist, meta) in enumerate(zip(ids, dists, metas), 1):
        row = conn.execute("SELECT work_id, char_start, char_end FROM chunk WHERE id=?", (cid,)).fetchone()
        excerpt = conn.execute(
            "SELECT substr(norm_text, ?+1, min(280, ?-?)) FROM work WHERE id=?",
            (row["char_start"], row["char_end"], row["char_start"], row["work_id"])
        ).fetchone()[0] or ""
        print(f"#{rank} dist={dist:.4f} id={cid} meta={dict(meta)}")
        print(excerpt.replace("\n", " ")[:280])
        print("-" * 80)

if __name__ == "__main__":
    main()
