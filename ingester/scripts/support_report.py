#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sqlite3, textwrap, sys
from typing import List, Dict

def fetch(conn, sql, args=()):
    cur = conn.execute(sql, args); cols=[c[0] for c in cur.description]
    return [dict(zip(cols,row)) for row in cur.fetchall()]

def table_exists(conn, name: str) -> bool:
    r = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone()
    return bool(r)

def scene_header(wtitle, author, sidx, s0, s1, fmt):
    t = f"### {wtitle} — Scene {sidx} [{s0}–{s1}]" if fmt=="md" else f"{wtitle} — Scene {sidx} [{s0}–{s1}]"
    a = f"by {author or '—'}"
    return t + ("\n" + a if fmt=="md" else f" — {a}")

def md_code(s): return "```\n" + s + "\n```"

def render_support(snips: List[str], fmt: str) -> str:
    if not snips:
        return "(no support snippets)"
    out=[]
    for i, t in enumerate(snips, 1):
        t = t.strip().replace("\n", " ")
        short = t if len(t)<=480 else t[:479]+"…"
        if fmt=="md":
            out.append(f"**{i}.** {short}")
        else:
            out.append(f"{i:>2}. {short}")
    return "\n".join(out)

def render_sanity(rows: List[dict], tropes_by_id: Dict[str,str], fmt: str) -> str:
    if not rows:
        return "(no trope_sanity rows; run judge-scenes with the new rerank_support changes)"
    if fmt=="md":
        out = ["| trope | lex | sem_sim | weight |", "|---|:--:|:------:|:-----:|"]
        for r in rows:
            out.append(f"| {tropes_by_id.get(r['trope_id'], r['trope_id'])} | {r['lex_ok']} | {r['sem_sim']:.3f} | {r['weight']:.2f} |")
        return "\n".join(out)
    else:
        lines=[]
        for r in rows:
            lines.append(f"- {tropes_by_id.get(r['trope_id'], r['trope_id'])}: "
                         f"lex={r['lex_ok']} sem_sim={r['sem_sim']:.3f} weight={r['weight']:.2f}")
        return "\n".join(lines)

def render_findings(rows: List[dict], fmt: str, threshold: float|None=None) -> str:
    if not rows:
        return "(no findings for this scene)"
    out=[]
    for r in rows:
        conf = float(r["confidence"] or 0.0)
        mark = "**" if (fmt=="md" and threshold is not None and conf >= threshold) else ""
        tro = r["trope"]
        span = f"[{r['evidence_start']}–{r['evidence_end']}]"
        rat  = (r["rationale"] or "").replace("\n"," ")
        rat  = rat if len(rat)<=120 else rat[:119]+"…"
        if fmt=="md":
            out.append(f"- {mark}{tro} ({conf:.2f}){mark} {span} — {rat}")
        else:
            out.append(f"- {tro} ({conf:.2f}) {span} — {rat}")
    return "\n".join(out)

def load_support_texts(conn, support_ids: List[str]) -> List[str]:
    if not support_ids:
        return []
    q = conn.execute(f"SELECT id,text,char_start,char_end,work_id FROM chunk WHERE id IN ({','.join(['?']*len(support_ids))})", tuple(support_ids))
    rows = {r["id"]: r for r in q.fetchall()}
    out=[]
    for cid in support_ids:
        r = rows.get(cid)
        if not r: continue
        txt = (r["text"] or "").strip()
        if txt:
            out.append(txt)
        else:
            w = conn.execute("SELECT norm_text FROM work WHERE id=?", (r["work_id"],)).fetchone()
            if w and w[0]:
                cs, ce = int(r["char_start"]), int(r["char_end"])
                out.append((w[0][cs:ce] or "").strip())
    return out

def report_for_scene(conn, scene_id: str, fmt: str="txt", threshold: float|None=None) -> str:
    s = fetch(conn, """
      SELECT s.id, s.idx, s.work_id, s.char_start, s.char_end, w.title, w.author, w.norm_text
      FROM scene s JOIN work w ON w.id=s.work_id WHERE s.id=?""", (scene_id,))
    if not s: return f"(scene {scene_id} not found)\n"
    S = s[0]
    s0, s1 = int(S["char_start"]), int(S["char_end"])
    head = scene_header(S["title"] or S["work_id"], S["author"], S["idx"], s0, s1, fmt)

    # support
    sup = fetch(conn, "SELECT support_ids,notes FROM scene_support WHERE scene_id=?", (scene_id,))
    support_texts=[]
    sup_notes=""
    if sup:
        try:
            ids = json.loads(sup[0]["support_ids"] or "[]")
            ids = ids if isinstance(ids, list) else []
        except Exception:
            ids = []
        support_texts = load_support_texts(conn, ids)
        sup_notes = sup[0].get("notes") or ""

    # sanity (optional table)
    tropes = fetch(conn, "SELECT id,name FROM trope")
    trope_name = {t["id"]:t["name"] for t in tropes}
    sanity_rows = []
    if table_exists(conn, "trope_sanity"):
        sanity_rows = fetch(conn, "SELECT trope_id,lex_ok,sem_sim,weight FROM trope_sanity WHERE scene_id=? ORDER BY trope_id", (scene_id,))

    # findings
    findings = fetch(conn, """
      SELECT f.trope_id, t.name AS trope, f.confidence, f.evidence_start, f.evidence_end, f.rationale, f.id
      FROM trope_finding f JOIN trope t ON t.id=f.trope_id
      WHERE f.scene_id=? ORDER BY f.confidence DESC, f.evidence_start
    """, (scene_id,))

    # assemble
    lines=[]
    lines.append(head)
    if fmt=="md": lines.append("")
    if sup_notes:
        lines.append(f"Rerank notes: {sup_notes}")
    lines.append( ("Support snippets:\n" + render_support(support_texts, fmt)) )
    lines.append("")
    lines.append( ("Trope sanity (lex/sem/weight):\n" + render_sanity(sanity_rows, trope_name, fmt)) )
    lines.append("")
    lines.append( ("Findings (post-weight):\n" + render_findings(findings, fmt, threshold=threshold)) )
    lines.append("")
    return "\n".join(lines)

def main():
    ap = argparse.ArgumentParser(description="Report chosen support, sanity weights, and findings per scene/work.")
    ap.add_argument("--db", required=True)
    ap.add_argument("--work-id")
    ap.add_argument("--scene-ids", help="comma-separated scene ids")
    ap.add_argument("--format", choices=["txt","md"], default="txt")
    ap.add_argument("--threshold", type=float)
    ap.add_argument("--out", help="output file; stdout if omitted")
    args = ap.parse_args()

    conn = sqlite3.connect(args.db); conn.row_factory = sqlite3.Row

    scene_ids=[]
    if args.scene_ids:
        scene_ids = [s.strip() for s in args.scene_ids.split(",") if s.strip()]
    elif args.work_id:
        r = fetch(conn, "SELECT id FROM scene WHERE work_id=? ORDER BY idx", (args.work_id,))
        scene_ids = [row["id"] for row in r]
    else:
        # all works → all scenes (be careful on big DBs)
        r = fetch(conn, "SELECT id FROM scene ORDER BY id")
        scene_ids = [row["id"] for row in r]

    blocks = []
    for sid in scene_ids:
        blocks.append(report_for_scene(conn, sid, fmt=args.format, threshold=args.threshold))

    doc = ("\n\n" if args.format=="md" else "\n\n").join(blocks)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(doc)
        print(f"✔ wrote {args.out}")
    else:
        sys.stdout.write(doc)

if __name__ == "__main__":
    main()
