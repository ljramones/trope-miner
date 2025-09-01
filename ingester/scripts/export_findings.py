#!/usr/bin/env python3
import argparse, csv, re, sqlite3, html
from pathlib import Path
from typing import List, Tuple, Dict, Any

# ---------- sentence helpers (matches span_verifier.py behavior) ----------
def sent_spans(text: str) -> List[Tuple[int, int]]:
    """Naive sentence splitter on ., !, ?, or double newlines.
    Returns trimmed (start,end) offsets into `text`.
    """
    spans, start = [], 0
    n = len(text)
    for m in re.finditer(r'[.!?]+(\s+|$)|\n{2,}', text):
        end = m.end()
        seg = text[start:end].strip()
        if seg:
            ls = len(text[start:end]) - len(text[start:end].lstrip())
            rs = len(text[start:end].rstrip())
            spans.append((start + ls, start + rs))
        start = end
    if start < n:
        tail = text[start:].strip()
        if tail:
            ls = len(text[start:]) - len(text[start:].lstrip())
            spans.append((start + ls, n))
    return spans or [(0, n)]

def sentence_for_span(text: str, sents: List[Tuple[int, int]], a: int, b: int) -> Tuple[int, int]:
    """Pick a sentence to represent evidence [a,b). Prefer the one containing the midpoint; fallback to nearest."""
    if not sents:
        return max(0, a), max(0, b)
    mid = (max(0, a) + max(0, b)) // 2
    for sa, sb in sents:
        if sa <= mid < sb:
            return sa, sb
    best, best_d = None, 10**12
    for sa, sb in sents:
        c = (sa + sb) // 2
        d = abs(c - mid)
        if d < best_d:
            best_d = d
            best = (sa, sb)
    return best if best else (max(0, a), max(0, b))

# ---------- DB fetch ----------
def fetch_work(conn: sqlite3.Connection, work_id: str) -> sqlite3.Row:
    conn.row_factory = sqlite3.Row
    w = conn.execute("SELECT id, title, author, norm_text FROM work WHERE id=?", (work_id,)).fetchone()
    if not w:
        raise SystemExit(f"work not found: {work_id}")
    return w

def fetch_rows(conn: sqlite3.Connection, work_id: str, limit: int) -> List[sqlite3.Row]:
    # Join scene/chapter to get scene_idx & chapter_idx alongside the finding.
    q = """
    SELECT
      f.id,
      f.work_id,
      f.scene_id,
      t.name                        AS trope,
      COALESCE(f.level,'')          AS level,
      COALESCE(f.confidence,0.0)    AS confidence,
      COALESCE(f.rationale,'')      AS rationale,
      COALESCE(f.evidence_start,0)  AS evidence_start,
      COALESCE(f.evidence_end,0)    AS evidence_end,
      COALESCE(f.created_at,'')     AS created_at,
      COALESCE(f.model,'')          AS model,
      s.idx                         AS scene_idx,
      c.idx                         AS chapter_idx
    FROM trope_finding f
    JOIN trope t     ON t.id = f.trope_id
    LEFT JOIN scene s   ON s.id = f.scene_id
    LEFT JOIN chapter c ON c.id = s.chapter_id
    WHERE f.work_id = ?
    ORDER BY f.created_at DESC
    """
    conn.row_factory = sqlite3.Row
    if limit and limit > 0:
        q += " LIMIT ?"
        return conn.execute(q, (work_id, limit)).fetchall()
    return conn.execute(q, (work_id,)).fetchall()

# ---------- writers ----------
def write_csv(out_path: Path, rows: List[Dict[str, Any]]) -> None:
    cols = [
        "id","work_id","scene_idx","chapter_idx","trope","level","confidence",
        "created_at","model","evidence_start","evidence_end","excerpt","evidence_sentence","rationale"
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})

def write_md(out_path: Path, rows: List[Dict[str, Any]], work_id: str) -> None:
    def trunc(s: str, n: int = 200) -> str:
        s = (s or "").replace("\n"," ").strip()
        return s if len(s) <= n else s[: n-1] + "…"
    with out_path.open("w", encoding="utf-8") as f:
        f.write(f"# Trope Findings for work `{work_id}`\n\n")
        f.write(f"Total findings: **{len(rows)}**\n\n")
        f.write("| Scene | Chap | Confidence | Trope | Level | Evidence sentence | Excerpt | Created | Model |\n")
        f.write("|---:|---:|---:|---|---|---|---|---|---|\n")
        for r in rows:
            f.write("| " + " | ".join([
                str(r.get("scene_idx","") if r.get("scene_idx") is not None else ""),
                str(r.get("chapter_idx","") if r.get("chapter_idx") is not None else ""),
                f"{float(r.get('confidence',0.0)):.2f}",
                (r.get("trope","") or "").replace("|","\\|"),
                (r.get("level","") or "").replace("|","\\|"),
                html.escape(trunc(r.get("evidence_sentence",""))),
                html.escape(trunc(r.get("excerpt",""))),
                (r.get("created_at","") or "").replace("|","\\|"),
                (r.get("model","") or "").replace("|","\\|"),
            ]) + " |\n")

# ---------- glue ----------
def build_rows(work: sqlite3.Row, findings: List[sqlite3.Row]) -> List[Dict[str, Any]]:
    text = work["norm_text"] or ""
    N = len(text)
    sents = sent_spans(text)
    out: List[Dict[str, Any]] = []

    for r in findings:
        s = int(r["evidence_start"] or 0)
        e = int(r["evidence_end"] or 0)
        # clamp & fix
        s = max(0, min(s, N)); e = max(0, min(e, N))
        if e < s: s, e = e, s
        excerpt = text[s:e] if e > s else ""

        sa, sb = sentence_for_span(text, sents, s, e)
        evidence_sentence = text[sa:sb].replace("\n"," ").strip()

        out.append({
            "id": r["id"],
            "work_id": r["work_id"],
            "scene_idx": r["scene_idx"],
            "chapter_idx": r["chapter_idx"],
            "trope": r["trope"],
            "level": r["level"],
            "confidence": float(r["confidence"] or 0.0),
            "created_at": r["created_at"],
            "model": r["model"],
            "evidence_start": s,
            "evidence_end": e,
            "excerpt": excerpt,
            "evidence_sentence": evidence_sentence,
            "rationale": r["rationale"],
        })
    return out

def main():
    ap = argparse.ArgumentParser(description="Export trope findings to CSV or Markdown (with scene/chapter & evidence sentence).")
    ap.add_argument("--db", required=True)
    ap.add_argument("--work-id", required=True)
    ap.add_argument("--format", choices=["csv","md"], default="csv")
    ap.add_argument("--out", required=True, help="Output file path")
    ap.add_argument("--limit", type=int, default=0, help="Max rows (0 = all)")
    args = ap.parse_args()

    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row

    work = fetch_work(conn, args.work_id)
    rows_db = fetch_rows(conn, args.work_id, args.limit)
    rows = build_rows(work, rows_db)

    if args.format == "csv":
        write_csv(out_path, rows)
    else:
        write_md(out_path, rows, args.work_id)

    conn.close()
    print(f"[wrote] {out_path.resolve()} ({len(rows)} rows)")

if __name__ == "__main__":
    main()
