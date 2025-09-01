#!/usr/bin/env python3
"""
HTML Report with Highlighted Evidence Spans

Generates a single self-contained HTML for one work, showing the full text with
<mark> highlights for each finding. Hover a highlight to see trope + confidence + rationale.

Enhancements:
- If the DB has `trope.tvtropes_url`, legend trope names link there (fallback to `trope.source_url`).
- If the DB has trope grouping tables (`trope_group`, `trope_group_member`), color by group;
  otherwise color per-trope.

Usage:
  python scripts/report_html.py --db tropes.db --work-id <WORK_ID> --out out/report_<WORK_ID>.html
"""
from __future__ import annotations

import argparse
import hashlib
import html
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Tuple, List


# ----------------------------- utilities -----------------------------

def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    cur = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1", (name,))
    return cur.fetchone() is not None

def _col_exists(conn: sqlite3.Connection, table: str, col: str) -> bool:
    cols = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
    return col in cols

def _pastel_rgba(key: str, alpha: float = 0.45) -> str:
    """
    Stable pastel-ish RGBA derived from a string key.
    """
    h = int(hashlib.md5(key.encode("utf-8")).hexdigest()[:6], 16)
    r = 180 + (h & 0x1F)
    g = 170 + ((h >> 5) & 0x1F)
    b = 160 + ((h >> 10) & 0x1F)
    return f"rgba({r % 255},{g % 255},{b % 255},{alpha:.2f})"


# ----------------------------- data fetch -----------------------------

def fetch(ctx):
    """
    Returns:
      work_row, findings_list

    findings_list items contain:
      s, e, confidence, rationale, trope, trope_id, trope_url (opt), trope_group (opt)
    """
    conn = sqlite3.connect(ctx.db)
    conn.row_factory = sqlite3.Row

    w = conn.execute(
        "SELECT id, title, author, norm_text FROM work WHERE id=?",
        (ctx.work_id,)
    ).fetchone()
    if not w:
        conn.close()
        raise SystemExit("work not found")

    # Feature-detect optional columns/tables
    has_tvt = _col_exists(conn, "trope", "tvtropes_url")
    has_src = _col_exists(conn, "trope", "source_url")
    groups_on = _table_exists(conn, "trope_group") and _table_exists(conn, "trope_group_member")

    # Base findings
    select_url_cols = []
    if has_tvt:
        select_url_cols.append("t.tvtropes_url AS tvt_url")
    if has_src:
        select_url_cols.append("t.source_url AS src_url")
    url_cols_sql = (", " + ", ".join(select_url_cols)) if select_url_cols else ""

    frows = conn.execute(f"""
        SELECT
          f.evidence_start AS s,
          f.evidence_end   AS e,
          f.confidence,
          f.rationale,
          t.id   AS trope_id,
          t.name AS trope
          {url_cols_sql}
        FROM trope_finding f
        JOIN trope t ON t.id = f.trope_id
        WHERE f.work_id = ?
        ORDER BY s ASC, e ASC
    """, (ctx.work_id,)).fetchall()

    # Optional group map: trope_id -> group_name (if present)
    group_by_trope: Dict[str, str] = {}
    if groups_on:
        grows = conn.execute("""
          SELECT m.trope_id AS trope_id, g.name AS group_name
          FROM trope_group_member m
          JOIN trope_group g ON g.id = m.group_id
        """).fetchall()
        for r in grows:
            tid = r["trope_id"]
            # If multiple groups per trope, stable-pick the first seen
            if tid not in group_by_trope:
                group_by_trope[tid] = r["group_name"]

    conn.close()

    # Normalize result rows into simple dicts with url + group
    out = []
    for r in frows:
        tvt = r["tvt_url"] if has_tvt else None
        src = r["src_url"] if has_src else None
        url = tvt or src or None
        out.append({
            "s": int(r["s"] or 0),
            "e": int(r["e"] or 0),
            "confidence": float(r["confidence"] or 0.0),
            "rationale": (r["rationale"] or ""),
            "trope": r["trope"],
            "trope_id": r["trope_id"],
            "trope_url": url,
            "trope_group": group_by_trope.get(r["trope_id"]),
        })
    return w, out


# ----------------------------- HTML assembly -----------------------------

def build_html(work, findings: List[dict], outpath: Path):
    text = work["norm_text"] or ""
    N = len(text)

    # Clamp and sort spans
    spans: List[Tuple[int, int, dict]] = []
    for r in findings:
        s, e = int(r["s"]), int(r["e"])
        if e <= s or s >= N:
            continue
        s = max(0, min(s, N))
        e = max(0, min(e, N))
        spans.append((s, e, r))
    spans.sort(key=lambda x: (x[0], x[1]))

    # Simple overlap policy: skip any span that starts before previous end
    merged: List[Tuple[int, int, dict]] = []
    cur_end = -1
    for s, e, r in spans:
        if s < cur_end:
            continue
        merged.append((s, e, r))
        cur_end = e
    spans = merged

    # Legend counts and metadata
    legend_info: Dict[str, dict] = {}
    for _, _, r in spans:
        t = r["trope"]
        if t not in legend_info:
            legend_info[t] = {
                "count": 0,
                "url": r.get("trope_url"),
                "group": r.get("trope_group")
            }
        legend_info[t]["count"] += 1

    # HTML stitching
    out_fragments: List[str] = []
    last = 0
    for s, e, r in spans:
        if s > last:
            out_fragments.append(html.escape(text[last:s]))

        # choose color: group color if available, else trope color
        group_name = r.get("trope_group")
        color_key = group_name if group_name else r["trope"]
        bg = _pastel_rgba(color_key, alpha=0.45)

        tip_bits = [r["trope"]]
        if group_name:
            tip_bits.append(f"[{group_name}]")
        tip_bits.append(f"conf={r['confidence']:.2f}")
        if r["rationale"]:
            tip_bits.append((r["rationale"][:240]).replace("\n", " "))
        tip = " | ".join(tip_bits)

        style = (
            f"background:{bg}; padding:0 .15em; border-radius:.25rem;"
            "box-decoration-break:clone;-webkit-box-decoration-break:clone;"
        )
        out_fragments.append(
            f'<mark title="{html.escape(tip)}" style="{style}">{html.escape(text[s:e])}</mark>'
        )
        last = e
    if last < N:
        out_fragments.append(html.escape(text[last:]))

    # Legend HTML
    def legend_row(name: str, info: dict) -> str:
        # swatch uses group color if present
        color_key = info.get("group") or name
        swatch = _pastel_rgba(color_key, alpha=0.55)
        badge = (
            f'<span style="display:inline-block;width:1em;height:1em;'
            f'background:{swatch};margin-right:.5em;border-radius:.2em;"></span>'
        )
        label = html.escape(name)
        url = info.get("url")
        if url:
            label = f'<a href="{html.escape(url)}" target="_blank" rel="noopener noreferrer">{label}</a>'
        count = info.get("count", 0)
        group = info.get("group")
        group_txt = f' <small style="color:#666;">[{html.escape(group)}]</small>' if group else ""
        return f"<li>{badge}{label}{group_txt} <small>×{count}</small></li>"

    legend_html = "".join(
        legend_row(t, legend_info[t]) for t in sorted(
            legend_info.keys(), key=lambda k: (-legend_info[k]["count"], k.lower())
        )
    ) or '<li><small>No findings</small></li>'

    # Basic doc shell
    html_doc = f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Trope Report — {html.escape(work['title'] or work['id'])}</title>
<style>
  :root {{
    --border:#eee;
    --text:#111;
    --muted:#666;
  }}
  body {{
    font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
    line-height:1.55; margin:0; color:var(--text);
    background:#fff;
  }}
  header {{
    padding: 12px 16px; border-bottom:1px solid var(--border);
    position:sticky; top:0; background:#fff; z-index:2;
  }}
  main {{ display:flex; gap:24px; padding:16px; }}
  aside {{ width:300px; max-width:33%; }}
  pre#text {{
    white-space:pre-wrap; word-wrap:break-word; margin:0; padding:0;
    background:#fff;
  }}
  legend ul {{ list-style:none; padding-left:0; margin: 8px 0 0; }}
  legend li {{ margin: 6px 0; font-size: 0.95em; }}
  small {{ color:var(--muted); }}
  a:link, a:visited {{ color:#0b5; text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}
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
    <legend><ul>{legend_html}</ul></legend>
    <p style="margin-top:12px;"><small>Tip: hover highlights to see confidence and rationale.</small></p>
  </aside>
  <section style="flex:1; min-width:0;">
    <pre id="text">{''.join(out_fragments)}</pre>
  </section>
</main>
</body></html>"""
    outpath.parent.mkdir(parents=True, exist_ok=True)
    outpath.write_text(html_doc, encoding="utf-8")
    print(f"[report] wrote {outpath.resolve()}")


# ----------------------------- CLI -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Generate HTML report with highlighted trope findings")
    ap.add_argument("--db", required=True)
    ap.add_argument("--work-id", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    w, f = fetch(args)
    build_html(w, f, Path(args.out))


if __name__ == "__main__":
    main()
