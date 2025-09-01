#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
    --model nomic-embed-text --threshold 0.32 --max-sentences 2 --alpha 0.7 --min-gain 0.05
"""
from __future__ import annotations

import argparse, os, re, sqlite3, math, requests
from typing import List, Tuple, Optional

# ------------------------ DB helpers ------------------------

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
    # Assume created_at exists in most DBs; if not, rowid fallback
    has_created = any(r[1] == "created_at" for r in conn.execute("PRAGMA table_info(trope_finding);"))
    order = "f.created_at ASC" if has_created else "f.rowid ASC"
    q = f"""
    SELECT f.id, f.scene_id, f.evidence_start, f.evidence_end, f.trope_id,
           t.name AS trope_name, COALESCE(t.summary,'') AS summary,
           s.char_start AS scene_start, s.char_end AS scene_end,
           w.norm_text
    FROM trope_finding f
    JOIN trope t  ON t.id = f.trope_id
    JOIN scene s  ON s.id = f.scene_id
    JOIN work  w  ON w.id = f.work_id
    WHERE f.work_id = ?
    ORDER BY {order}
    """
    conn.row_factory = sqlite3.Row
    return conn.execute(q, (work_id,)).fetchall()

# ------------------------ Text utils ------------------------

def sent_spans(text: str) -> List[Tuple[int,int]]:
    """Very simple sentence splitter on ., !, ?, or multiple newlines. Returns trimmed (start,end) pairs."""
    spans, start = [], 0
    n = len(text)
    for m in re.finditer(r'[.!?]+(?:\s+|$)|\n{2,}', text):
        end = m.end()
        seg = text[start:end]
        # trim leading/trailing whitespace while keeping absolute offsets
        ls = len(seg) - len(seg.lstrip())
        rs = len(seg.rstrip())
        if rs > ls:
            spans.append((start + ls, start + rs))
        start = end
    if start < n:
        tail = text[start:]
        ls = len(tail) - len(tail.lstrip())
        rs = len(tail.rstrip())
        if rs > ls:
            spans.append((start + ls, start + rs))
    return spans or [(0, n)]

def clip(a:int,b:int,n:int)->Tuple[int,int]:
    return max(0,min(a,n)), max(0,min(b,n))

def uniq_spans(spans: List[Tuple[int,int]]) -> List[Tuple[int,int]]:
    seen = set(); out=[]
    for s in spans:
        if s not in seen:
            seen.add(s); out.append(s)
    return out

# ------------------------ Similarity ------------------------

def cosine(a: List[float], b: List[float]) -> float:
    if not a or not b: return 0.0
    if len(a) != len(b): return 0.0
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
    # Try both payload styles for compatibility
    e = _post({"model": model, "input": text}) or _post({"model": model, "prompt": text})
    if not e:
        raise RuntimeError("empty embedding")
    return e

# ------------------------ Flags ------------------------

NEG_RE = re.compile(r'\b(no|not|never|without|hardly|rarely|seldom|none)\b', re.I)

def negation_cue(text: str) -> bool:
    # cheap heuristic: any negation token inside the candidate span
    return bool(NEG_RE.search(text))

# ------------------------ Main pass ------------------------

def main():
    ap = argparse.ArgumentParser(description="Tighten/flag finding spans with embedding similarity.")
    ap.add_argument("--db", required=True)
    ap.add_argument("--work-id", required=True)
    ap.add_argument("--ollama-url", default=os.getenv("OLLAMA_BASE_URL","http://localhost:11434"))
    ap.add_argument("--model", default="nomic-embed-text")
    ap.add_argument("--threshold", type=float, default=0.32, help="min acceptable verifier score (0..1)")
    ap.add_argument("--alpha", type=float, default=0.7, help="weight on sim(span,trope_def) vs sim(span,scene)")
    ap.add_argument("--min-gain", type=float, default=0.05, help="only replace span if best_score >= orig_score + min_gain")
    ap.add_argument("--max-sentences", type=int, default=2, help="expand search ±N sentences")
    ap.add_argument("--max-chars", type=int, default=280, help="cap span length when embedding")
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
        work_text  = r["norm_text"] or ""
        s0, s1     = int(r["scene_start"]), int(r["scene_end"])
        scene_txt  = work_text[s0:s1]
        if not scene_txt:
            continue

        # current span (absolute -> scene-local)
        e0, e1 = r["evidence_start"], r["evidence_end"]
        if e0 is None or e1 is None:
            e0, e1 = s0, s1
        e0, e1 = int(e0), int(e1)
        e0, e1 = clip(e0, e1, len(work_text))
        e0s, e1s = max(0, e0 - s0), max(0, e1 - s0)
        e0s, e1s = clip(e0s, e1s, len(scene_txt))
        if e1s <= e0s:
            e0s, e1s = 0, min(len(scene_txt), args.max_chars)

        # build candidate windows from sentence boundaries near the original
        spans = sent_spans(scene_txt)
        # find a sentence that intersects the original
        idx = max(0, min(len(spans)-1, next((i for i,(a,b) in enumerate(spans) if not (e1s<=a or e0s>=b)), 0)))

        # Generate candidates by expanding ±k sentences around idx; also include the exact original span
        cand_spans: List[Tuple[int,int]] = [(e0s, e1s)]
        for k in range(-args.max_sentences, args.max_sentences+1):
            j = max(0, min(len(spans)-1, idx + k))
            a = min(spans[idx][0], spans[j][0])
            b = max(spans[idx][1], spans[j][1])
            cand_spans.append((a, b))

        # Cap by max-chars (centered around the original midpoint)
        mid = (e0s + e1s)//2
        capped: List[Tuple[int,int]] = []
        for a,b in cand_spans:
            if b - a <= args.max_chars:
                capped.append((a,b))
            else:
                half = args.max_chars // 2
                na = max(0, min(mid - half, len(scene_txt)-args.max_chars))
                nb = na + args.max_chars
                capped.append((na, nb))

        cand_spans = uniq_spans([clip(a,b,len(scene_txt)) for (a,b) in capped])

        trope_text = f"{r['trope_name']}. {r['summary']}".strip()
        try:
            trope_emb = embed(args.ollama_url, args.model, trope_text[:1024])
            scene_emb = embed(args.ollama_url, args.model, scene_txt[:4096])
        except Exception as ex:
            print(f"[warn] embedding failed for trope/scene (finding={r['id'][:8]}): {ex}")
            continue

        def score_text(txt: str) -> float:
            try:
                emb = embed(args.ollama_url, args.model, txt)
            except Exception:
                return 0.0
            s_td = cosine(emb, trope_emb)
            s_sc = cosine(emb, scene_emb)
            return args.alpha * s_td + (1.0 - args.alpha) * s_sc

        # Score original first
        orig_text = scene_txt[e0s:e1s][:args.max_chars]
        orig_score = score_text(orig_text)

        best_span = (e0s, e1s)
        best_score = orig_score
        best_text = orig_text

        for a,b in cand_spans:
            txt = scene_txt[a:b][:args.max_chars]
            if not txt or (a,b) == (e0s,e1s):
                continue
            sc = score_text(txt)
            if sc > best_score:
                best_score, best_span, best_text = sc, (a,b), txt

        # Decide flag
        flag = "ok"
        if best_score < args.threshold:
            flag = "low_sim"
        elif negation_cue(best_text):
            flag = "negation_cue"

        # Decide whether to adopt the best span
        adopt = False
        if best_span != (e0s, e1s) and best_score >= orig_score + args.min_gain:
            adopt = True
        # If original is below threshold but best meets threshold, adopt regardless of gain
        if best_span != (e0s, e1s) and orig_score < args.threshold <= best_score:
            adopt = True

        new_e0_abs, new_e1_abs = e0, e1
        if adopt:
            new_e0_abs, new_e1_abs = s0 + best_span[0], s0 + best_span[1]

        if args.dry_run:
            changed = "UPDATED" if adopt else "keep"
            print(f"{r['id'][:8]} scene={r['scene_id'][:8]} trope={r['trope_id'][:8]} "
                  f"orig=({e0}-{e1}) score={orig_score:.3f} "
                  f"best=({new_e0_abs}-{new_e1_abs}) score={best_score:.3f} flag={flag} {changed}")
            continue

        # Write back
        try:
            with conn:
                conn.execute(
                    "UPDATE trope_finding SET evidence_start=?, evidence_end=?, verifier_score=?, verifier_flag=? WHERE id=?",
                    (int(new_e0_abs), int(new_e1_abs), float(best_score), flag, r["id"])
                )
            if adopt:
                updated += 1
        except Exception as ex:
            print(f"[warn] DB update failed for {r['id'][:8]}: {ex}")

    if not args.dry_run:
        print(f"[done] updated spans: {updated} (of {len(rows)})")

if __name__ == "__main__":
    main()
