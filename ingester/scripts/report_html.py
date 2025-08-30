#!/usr/bin/env python3
"""
HTML Report with Highlighted Evidence Spans

Generates a single self-contained HTML for one work, showing the full text with
<mark> highlights for each finding. Hover a highlight to see trope + confidence + rationale.

Usage:
  python scripts/report_html.py --db tropes.db --work-id <WORK_ID> --out out/report_<WORK_ID>.html
"""
from __future__ import annotations

import argparse, html, hashlib, random, sqlite3
from pathlib import Path
from collections import defaultdict

def color_for(name: str) -> str:
    # stable pastel-ish color from name
    h = int(hashlib.md5(name.encode('utf-8')).hexdigest()[:6], 16)
    r = 200 + (h & 0x1F)      ; g = 180 + ((h>>5) & 0x1F) ; b = 170 + ((h>>10) & 0x1F)
    return f"rgba({r%255},{g%255},{b%255},0.45)"

def fetch(ctx):
    conn = sqlite3.connect(ctx.db)
    conn.row_factory = sqlite3.Row
    w = conn.execute("SELECT id,title,author,norm_text FROM work WHERE id=?", (ctx.work_id,)).fetchone()
    if not w: raise SystemExit("work not found")
    frows = conn.execute("""
        SELECT f.evidence_start s, f.evidence_end e, f.confidence, f.rationale,
               t.name AS trope
        FROM trope_finding f
        JOIN trope t ON t.id=f.trope_id
        WHERE f.work_id=?
        ORDER BY s ASC, e ASC
    """, (ctx.work_id,)).fetchall()
    conn.close()
    return w, frows

def build_html(work, findings, outpath: Path):
    text = work["norm_text"]
    N = len(text)
    # coalesce / clamp / skip overlaps
    spans = []
    for r in findings:
        s, e = int(r["s"] or 0), int(r["e"] or 0)
        if e <= s or s>=N: continue
        s = max(0, min(s, N)); e = max(0, min(e, N))
        spans.append((s, e, r))
    spans.sort(key=lambda x:(x[0], x[1]))

    merged = []
    cur_end = -1
    for s,e,r in spans:
        if s < cur_end:  # simple overlap policy: skip
            continue
        merged.append((s,e,r)); cur_end = e
    spans = merged

    # stitch HTML
    out = []
    last = 0
    counts = defaultdict(int)
    for s,e,r in spans:
        if s>last:
            out.append(html.escape(text[last:s]))
        trope = r["trope"]
        counts[trope]+=1
        tip = f"{trope} | conf={float(r['confidence']):.2f}\n{(r['rationale'] or '')[:240]}"
        style=f"background:{color_for(trope)}; padding:0 .15em; border-radius:.25rem;"
        out.append(f'<mark title="{html.escape(tip)}" style="{style}">{html.escape(text[s:e])}</mark>')
        last = e
    if last<N:
        out.append(html.escape(text[last:]))

    legend = "".join(
        f'<li><span style="display:inline-block;width:1em;height:1em;background:{color_for(t)};margin-right:.5em;border-radius:.2em;"></span>'
        f'{html.escape(t)} <small>×{c}</small></li>'
        for t,c in sorted(counts.items(), key=lambda kv:(-kv[1], kv[0]))
    )

    html_doc = f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Trope Report — {html.escape(work['title'] or work['id'])}</title>
<style>
body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; line-height:1.55; margin:0; }}
header {{ padding: 12px 16px; border-bottom:1px solid #eee; position:sticky; top:0; background:#fff; z-index:2; }}
main {{ display:flex; gap:24px; padding:16px; }}
aside {{ width:280px; max-width:33%; }}
pre#text {{ white-space:pre-wrap; word-wrap:break-word; margin:0; background:#fff; padding:0; }}
legend ul {{ list-style:none; padding-left:0; margin: 8px 0 0; }}
legend li {{ margin: 6px 0; font-size: 0.95em; }}
small {{ color:#666; }}
</style>
</head>
<body>
<header>
  <strong>{html.escape(work['title'] or '')}</strong>
  <span>— {html.escape(work['author'] or '')}</span>
</header>
<main>
  <aside>
    <h3 style="margin:0 0 .25rem 0;">Legend</h3>
    <legend><ul>{legend or '<li><small>No findings</small></li>'}</ul></legend>
  </aside>
  <section style="flex:1; min-width:0;">
    <pre id="text">{''.join(out)}</pre>
  </section>
</main>
</body></html>"""
    outpath.parent.mkdir(parents=True, exist_ok=True)
    outpath.write_text(html_doc, encoding="utf-8")
    print(f"[report] wrote {outpath.resolve()}")

def main():
    ap = argparse.ArgumentParser(description="Generate HTML report with highlighted trope findings")
    ap.add_argument("--db", required=True)
    ap.add_argument("--work-id", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    w, f = fetch(args)
    build_html(w, f, Path(args.out))

if __name__ == "__main__":
    from pathlib import Path
    main()
