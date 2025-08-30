#!/usr/bin/env python3
import argparse, csv, sqlite3
from pathlib import Path
from typing import Optional, List

def fetch_rows(conn: sqlite3.Connection, work_id: str, limit: int) -> List[sqlite3.Row]:
    # Build excerpt directly from work.norm_text using absolute offsets.
    q = """
    SELECT
      f.id,
      f.work_id,
      t.name                        AS trope,
      COALESCE(f.level,'')          AS level,
      COALESCE(f.confidence,0.0)    AS confidence,
      COALESCE(f.rationale,'')      AS rationale,
      COALESCE(f.evidence_start,0)  AS evidence_start,
      COALESCE(f.evidence_end,0)    AS evidence_end,
      COALESCE(f.created_at,'')     AS created_at,
      COALESCE(f.model,'')          AS model,
      CASE
        WHEN f.evidence_end > f.evidence_start
        THEN substr(w.norm_text,
                    f.evidence_start + 1,                -- sqlite substr is 1-based
                    (f.evidence_end - f.evidence_start)  -- length
        )
        ELSE ''
      END AS excerpt
    FROM trope_finding f
    JOIN trope t ON t.id = f.trope_id
    JOIN work  w ON w.id = f.work_id
    WHERE f.work_id = ?
    ORDER BY f.created_at DESC
    """
    if limit and limit > 0:
        q += " LIMIT ?"
        return conn.execute(q, (work_id, limit)).fetchall()
    return conn.execute(q, (work_id,)).fetchall()

def write_csv(out_path: Path, rows: List[sqlite3.Row]) -> None:
    cols = ["id","work_id","trope","level","confidence","created_at","model",
            "evidence_start","evidence_end","excerpt","rationale"]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r[c] for c in cols})

def write_md(out_path: Path, rows: List[sqlite3.Row], work_id: str) -> None:
    with out_path.open("w", encoding="utf-8") as f:
        f.write(f"# Trope Findings for work `{work_id}`\n\n")
        f.write(f"Total findings: **{len(rows)}**\n\n")
        f.write("| Confidence | Trope | Level | Excerpt | Created | Model |\n")
        f.write("|---:|---|---|---|---|---|\n")
        for r in rows:
            conf = f"{float(r['confidence']):.2f}"
            trope = (r["trope"] or "").replace("|","\\|")
            level = (r["level"] or "").replace("|","\\|")
            excerpt = (r["excerpt"] or "").replace("\n"," ").replace("|","\\|").strip()
            created = (r["created_at"] or "").replace("|","\\|")
            model = (r["model"] or "").replace("|","\\|")
            f.write(f"| {conf} | {trope} | {level} | {excerpt} | {created} | {model} |\n")

def main():
    ap = argparse.ArgumentParser(description="Export trope findings to CSV or Markdown.")
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

    rows = fetch_rows(conn, args.work_id, args.limit)
    if args.format == "csv":
        write_csv(out_path, rows)
    else:
        write_md(out_path, rows, args.work_id)

    conn.close()
    print(f"[wrote] {out_path.resolve()} ({len(rows)} rows)")

if __name__ == "__main__":
    main()
