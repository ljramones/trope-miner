# scripts/verify_spans.py
import argparse, os, re, sqlite3, math, requests, json
from pathlib import Path
import chromadb

def cosine(a,b):
    num = sum(x*y for x,y in zip(a,b))
    da = math.sqrt(sum(x*x for x in a)); db = math.sqrt(sum(y*y for y in b))
    return 0.0 if da==0 or db==0 else num/(da*db)

def embed(texts, base, model):
    url = f"{base.rstrip('/')}/api/embeddings"
    out=[]
    for t in texts:
        r = requests.post(url, json={"model": model, "prompt": t})
        r.raise_for_status()
        out.append(r.json()["embedding"])
    return out

def sentence_bounds(s):
    # simple sentence tokenizer: split on .,?!— and line breaks keeping spans
    # we return indices of sentence boundaries
    seps = re.finditer(r'([.!?]|—)+\s+', s)
    idx=[0]
    for m in seps:
        idx.append(m.end())
    if idx[-1] != len(s): idx.append(len(s))
    return idx

def snap_to_sentence(scene_text, abs_start, abs_end, scene_abs_start):
    # convert to scene-relative
    s = max(0, abs_start - scene_abs_start)
    e = max(0, abs_end   - scene_abs_start)
    s = min(s, len(scene_text)); e = min(max(e,s+1), len(scene_text))

    bounds = sentence_bounds(scene_text)
    # find nearest left boundary <= s, and right boundary >= e
    left  = max([b for b in bounds if b <= s] or [0])
    right = min([b for b in bounds if b >= e] or [len(scene_text)])
    return (scene_abs_start + left, scene_abs_start + right)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--db', required=True)
    ap.add_argument('--work-id', required=True)
    ap.add_argument('--apply', action='store_true')
    ap.add_argument('--delta', type=float, default=0.05, help="min similarity gain to adopt snapped span")
    ap.add_argument('--topk', type=int, default=3, help="top-K chunks for local coherence")
    args = ap.parse_args()

    DB = sqlite3.connect(args.db); DB.row_factory = sqlite3.Row

    # env
    OLLAMA = os.getenv("OLLAMA_BASE_URL","http://127.0.0.1:11434")
    EMBED_MODEL = os.getenv("EMBED_MODEL","nomic-embed-text")
    CHOST = os.getenv("CHROMA_HOST","127.0.0.1"); CPORT = int(os.getenv("CHROMA_PORT","8000"))
    TROPE_COL = os.getenv("TROPE_COLLECTION","trope-catalog-nomic-cos")
    CHUNK_COL = os.getenv("CHUNK_COLLECTION","trope-miner-v1-cos")

    cl = chromadb.HttpClient(host=CHOST, port=CPORT)
    tropes = cl.get_or_create_collection(TROPE_COL, metadata={"hnsw:space":"cosine"})
    chunks = cl.get_or_create_collection(CHUNK_COL, metadata={"hnsw:space":"cosine"})

    # pull work + text
    work = DB.execute("SELECT id,title,author,norm_text FROM work WHERE id=?", (args.work_id,)).fetchone()
    if not work: raise SystemExit("work not found")

    # scenes map
    scenes = {r["id"]:r for r in DB.execute("SELECT id,idx,char_start,char_end FROM scene WHERE work_id=?", (work["id"],))}
    # findings
    q = DB.execute("""
      SELECT f.id,f.scene_id,f.trope_id,f.evidence_start AS s,f.evidence_end AS e,t.name AS trope
      FROM trope_finding f JOIN trope t ON t.id=f.trope_id
      WHERE f.work_id=? ORDER BY f.scene_id,f.evidence_start
    """, (work["id"],))

    rows = q.fetchall()
    print(f"# findings: {len(rows)}")

    for r in rows:
      srec = scenes[r["scene_id"]]
      scene_text = work["norm_text"][srec["char_start"]:srec["char_end"]]
      span_text = work["norm_text"][r["s"]:r["e"]]

      # embeddings
      span_emb = embed([span_text], OLLAMA, EMBED_MODEL)[0]

      # trope embedding (id = trope_id stored as document id in trope collection)
      trope_vec = None
      try:
          g = tropes.get(ids=[r["trope_id"]], include=["embeddings"])
          if g["embeddings"] and len(g["embeddings"])==1:
              trope_vec = g["embeddings"][0]
      except Exception:
          pass

      trope_sim = cosine(span_emb, trope_vec) if trope_vec is not None else None

      # local coherence: query top-K nearest chunks
      # note: collection must contain chunk embeddings for this work; we filter by metadata if present
      top = chunks.query(query_embeddings=[span_emb], n_results=args.topk)
      local_sim = None
      if top and top.get("distances"):
          # Chroma returns distances; for cosine space, distance ~ (1 - cosine). Convert.
          ds = top["distances"][0]
          local_sim = sum(1.0 - d for d in ds)/len(ds)

      # sentence snap
      ns, ne = snap_to_sentence(scene_text, r["s"], r["e"], srec["char_start"])
      if (ns,ne) != (r["s"], r["e"]):
          snapped_text = work["norm_text"][ns:ne]
          snapped_emb = embed([snapped_text], OLLAMA, EMBED_MODEL)[0]
          snapped_trope_sim = cosine(snapped_emb, trope_vec) if trope_vec is not None else None
          snapped_local_top = chunks.query(query_embeddings=[snapped_emb], n_results=args.topk)
          snapped_local_sim = None
          if snapped_local_top and snapped_local_top.get("distances"):
              ds = snapped_local_top["distances"][0]
              snapped_local_sim = sum(1.0 - d for d in ds)/len(ds)

          # choose metric: prefer trope alignment; fall back to local if trope_vec missing
          base = trope_sim if trope_sim is not None else (local_sim or 0.0)
          cand = snapped_trope_sim if snapped_trope_sim is not None else (snapped_local_sim or 0.0)

          if cand - base >= args.delta:
              print(f"[SUGGEST] {r['id']} {r['trope']} {r['s']}–{r['e']} -> {ns}–{ne}  (Δ={cand-base:.3f})")
              if args.apply:
                  # history row
                  DB.execute("""
                    INSERT INTO trope_finding_human
                      (id,finding_id,decision,corrected_start,corrected_end,corrected_trope_id,note,reviewer)
                    VALUES (lower(hex(randomblob(16))), ?, 'edit', ?, ?, NULL, 'auto-snap', 'verifier')
                  """, (r["id"], ns, ne))
                  DB.execute("UPDATE trope_finding SET evidence_start=?, evidence_end=? WHERE id=?", (ns, ne, r["id"]))
                  DB.commit()
          else:
              print(f"[KEEP]    {r['id']} {r['trope']} {r['s']}–{r['e']} (no gain)")
      else:
          print(f"[OK]      {r['id']} {r['trope']} {r['s']}–{r['e']}")

if __name__ == "__main__":
    main()
