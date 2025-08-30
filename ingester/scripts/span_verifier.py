#!/usr/bin/env python3
"""
Span Verifier
-------------
Refines LLM evidence spans and adds a verifier score/flag.

Heuristic:
- Snap spans to nearby sentence boundaries (within N sentences).
- Score candidate span with cosine(sim(span, trope_text)) and sim(span, scene_text).
- Pick the best; if score < threshold, flag 'low_sim'. Also flag simple negation cues.

Writes back:
- trope_finding.evidence_start / evidence_end (possibly adjusted)
- trope_finding.verifier_score (REAL)
- trope_finding.verifier_flag  (TEXT: 'ok' | 'low_sim' | 'negation_cue')

Usage:
  python scripts/span_verifier.py --db tropes.db --work-id <WORK_ID> \
    --model nomic-embed-text --threshold 0.25 --max-sentences 2
"""
from __future__ import annotations

import argparse, os, re, sqlite3, math, json, requests
from typing import List, Tuple, Optional

def ensure_columns(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cols = {r[1] for r in cur.execute("PRAGMA table_info(trope_finding);")}
    updates = []
    if "verifier_score" not in cols:
        updates.append("ALTER TABLE trope_finding ADD COLUMN verifier_score REAL;")
    if "verifier_flag" not in cols:
        updates.append("ALTER TABLE trope_finding ADD COLUMN verifier_flag TEXT;")
    for sql in updates:
        cur.execute(sql)
    if updates:
        conn.commit()

def fetch_findings(conn: sqlite3.Connection, work_id: str) -> List[sqlite3.Row]:
    q = """
    SELECT f.id, f.scene_id, f.evidence_start, f.evidence_end, f.trope_id,
           t.name AS trope_name, COALESCE(t.summary,'') AS summary,
           s.char_start AS scene_start, s.char_end AS scene_end,
           w.norm_text
    FROM trope_finding f
    JOIN trope t  ON t.id = f.trope_id
    JOIN scene s  ON s.id = f.scene_id
    JOIN work  w  ON w.id = f.work_id
    WHERE f.work_id = ?
    ORDER BY f.created_at ASC
    """
    conn.row_factory = sqlite3.Row
    return conn.execute(q, (work_id,)).fetchall()

def sent_spans(text: str) -> List[Tuple[int,int]]:
    """Very simple sentence splitter on ., !, ?, or double newlines."""
    spans, start = [], 0
    n = len(text)
    for m in re.finditer(r'[.!?]+(\s+|$)|\n{2,}', text):
        end = m.end()
        seg = text[start:end].strip()
        if seg:
            # trim leading/trailing whitespace in span mapping
            ls = len(text[start:end]) - len(text[start:end].lstrip())
            rs = len(text[start:end].rstrip())
            spans.append((start+ls, start+rs))
        start = end
    if start < n:
        tail = text[start:].strip()
        if tail:
            ls = len(text[start:]) - len(text[start:].lstrip())
            spans.append((start+ls, n))
    return spans or [(0, n)]

def clip(a:int,b:int,n:int)->Tuple[int,int]:
    return max(0,min(a,n)), max(0,min(b,n))

def cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a)!=len(b): return 0.0
    dp = sum(x*y for x,y in zip(a,b))
    na = math.sqrt(sum(x*x for x in a)); nb = math.sqrt(sum(y*y for y in b))
    if na==0 or nb==0: return 0.0
    return dp/(na*nb)

def embed(ollama_url: str, model: str, text: str, timeout: int=90) -> List[float]:
    url = ollama_url.rstrip('/') + "/api/embeddings"
    def _post(payload):
        r = requests.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        emb = data.get("embedding")
        if emb is None and isinstance(data.get("data"), list) and data["data"]:
            emb = data["data"][0].get("embedding")
        if emb is None and isinstance(data.get("embeddings"), list) and data["embeddings"]:
            first = data["embeddings"][0]
            if isinstance(first, list): emb = first
        return emb or []
    e = _post({"model": model, "input": text}) or _post({"model": model, "prompt": text})
    if not e: raise RuntimeError("empty embedding")
    return e

NEG_RE = re.compile(r'\b(no|not|never|without|hardly|rarely)\b', re.I)

def main():
    ap = argparse.ArgumentParser(description="Tighten/flag finding spans with embedding similarity.")
    ap.add_argument("--db", required=True)
    ap.add_argument("--work-id", required=True)
    ap.add_argument("--ollama-url", default=os.getenv("OLLAMA_BASE_URL","http://localhost:11434"))
    ap.add_argument("--model", default="nomic-embed-text")
    ap.add_argument("--threshold", type=float, default=0.25, help="min acceptable verifier score")
    ap.add_argument("--max-sentences", type=int, default=2, help="expand search ±N sentences")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    ensure_columns(conn)

    rows = fetch_findings(conn, args.work_id)
    if not rows:
        print("[info] no findings to verify")
        return

    updated = 0
    for r in rows:
        wtxt      = r["norm_text"]
        s0, s1    = int(r["scene_start"]), int(r["scene_end"])
        scene_txt = wtxt[s0:s1]

        e0, e1 = r["evidence_start"], r["evidence_end"]
        if e0 is None or e1 is None:
            e0, e1 = s0, s1
        e0, e1 = int(e0), int(e1)
        e0, e1 = clip(e0, e1, len(wtxt))
        # map to scene-local
        e0s, e1s = max(0, e0 - s0), max(0, e1 - s0)
        e0s, e1s = clip(e0s, e1s, len(scene_txt))

        spans = sent_spans(scene_txt)
        # find sentence covering original span
        idx = 0
        for i,(a,b) in enumerate(spans):
            if a <= e0s < b or a < e1s <= b or (e0s<=a and e1s>=b):
                idx = i; break

        # candidates: sentence, ±neighbors up to max-sentences
        cands = []
        for k in range(-args.max_sentences, args.max_sentences+1):
            a = max(0, min(idx, len(spans)-1))
            j = max(0, min(a+k, len(spans)-1))
            start = min(spans[a][0], spans[j][0])
            end   = max(spans[a][1], spans[j][1])
            cands.append((start, end))
        #
