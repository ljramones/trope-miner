# scripts/report_highlights.py
import argparse, html, os, sqlite3, re
from pathlib import Path

CSS = """
body{font:15px/1.5 -apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial,sans-serif;color:#111;margin:0;background:#fff}
header{position:sticky;top:0;background:#fff;border-bottom:1px solid #eee;padding:10px 16px;z-index:10}
.container{max-width:1000px;margin:0 auto;padding:16px}
.scene{border:1px solid #eee;margin:16px 0;border-radius:6px;overflow:hidden}
.scene h3{margin:0;padding:10px 12px;background:#fafafa;border-bottom:1px solid #eee;font-weight:600}
.scene pre{margin:0;padding:12px 12px 16px;white-space:pre-wrap}
mark{background:#fff59d;border-radius:2px;padding:0 .1em}
.legend{display:flex;flex-wrap:wrap;gap:.5em;margin-top:8px}
.badge{display:inline-block;padding:.1em .4em;border-radius:3px;background:#eee}
.small{opacity:.7;font-size:.9em}
"""

JS = """
document.addEventListener('click', (e)=>{
  const a = e.target.closest('[data-jump]');
  if(!a) return;
  e.preventDefault();
  const id = a.getAttribute('data-jump');
  const el = document.querySelector(`[data-span-id="${id}"]`);
  if(el){ el.scrollIntoView({behavior:'smooth',block:'center'}); el.classList.add('pulse'); setTimeout(()=>el.classList.remove('pulse'),800); }
});
"""

def sanitize(title: str) -> str:
    return re.sub(r'[^A-Za-z0-9._-]+','_', title.strip()) or "work"

def wrap_with_marks(text: str, spans):
    # spans: list of dicts with s,e, id, trope
    # assumes s/e are scene-relative code-point indices
    spans = sorted([s for s in spans if s['e']>s['s']], key=lambda x:(x['s'], x['e']))
    out=[]; pos=0
    for sp in spans:
        s,e = sp['s'], sp['e']
        if s>pos: out.append(html.escape(text[pos:s]))
        out.append(f'<mark data-span-id="{html.escape(sp["id"])}" title="{html.escape(sp["trope"])}">{html.escape(text[s:e])}</mark>')
        pos=e
    out.append(html.escape(text[pos:]))
    return ''.join(out)

def fetch(conn, sql, args=()):
    cur = conn.execute(sql, args); cols=[c[0] for c in cur.description]
    return [dict(zip(cols,row)) for row in cur.fetchall()]

def build_for_work(conn, work):
    wid = work['id']
    # scenes
    scenes = fetch(conn, "SELECT id,idx,char_start,char_end FROM scene WHERE work_id=? ORDER BY idx", (wid,))
    # findings
    frows = fetch(conn, """
      SELECT f.id,f.scene_id,f.evidence_start AS s,f.evidence_end AS e, t.name AS trope
      FROM trope_finding f JOIN trope t ON t.id=f.trope_id
      WHERE f.work_id=?
      ORDER BY f.scene_id,f.evidence_start,f.evidence_end
    """, (wid,))
    # text
    wrow = fetch(conn, "SELECT norm_text,title,author FROM work WHERE id=?", (wid,))[0]
    text, title, author = wrow['norm_text'], wrow.get('title') or wid, wrow.get('author') or '—'

    # group findings by scene
    by_scene={}
    for f in frows:
        by_scene.setdefault(f['scene_id'],[]).append(f)

    # assemble HTML
    parts=[f"<!doctype html><meta charset='utf-8'><title>Highlights — {html.escape(title)}</title>",
           f"<style>{CSS}</style><header><div class='container'><h1 style='margin:0'>Highlights — {html.escape(title)}</h1><div class='small'>{html.escape(author)}</div></div></header>",
           "<div class='container'>",
           "<div class='legend small'><span class='badge'>Click any finding in the list below to jump to its highlight.</span></div>"]

    # flat list of findings (jump links)
    parts.append("<h2>Findings</h2><ol>")
    for f in frows:
        parts.append(f"<li><a href='#' data-jump='{f['id']}'>{html.escape(f['trope'])}</a> "
                     f"<span class='small'>(scene {next((s['idx'] for s in scenes if s['id']==f['scene_id']), '?')}, {f['s']}–{f['e']})</span></li>")
    parts.append("</ol>")

    # scenes with highlights
    for s in scenes:
        s_id=s['id']; start=s['char_start']; end=s['char_end']
        scene_text = text[start:end]
        # convert absolute → scene-relative
        spans = [{'id':f['id'], 's':max(0,f['s']-start),'e':max(0,f['e']-start),'trope':f['trope']} for f in by_scene.get(s_id,[])]
        html_body = wrap_with_marks(scene_text, spans)
        parts.append(f"<div class='scene'><h3>Scene {s['idx']} <span class='small'>[{start}–{end}]</span></h3><pre>{html_body}</pre></div>")

    parts.append(f"<script>{JS}</script></div>")
    return '\n'.join(parts)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--db', required=True)
    ap.add_argument('--work-id')
    ap.add_argument('--title')  # exact match
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row

    if args.work_id:
        works = fetch(conn, "SELECT id,title,author FROM work WHERE id=?", (args.work_id,))
    elif args.title:
        works = fetch(conn, "SELECT id,title,author FROM work WHERE title=?", (args.title,))
    else:
        works = fetch(conn, "SELECT id,title,author FROM work ORDER BY created_at DESC")

    outdir = Path("reports"); outdir.mkdir(parents=True, exist_ok=True)

    for w in works:
        html_doc = build_for_work(conn, w)
        name = sanitize(w.get('title') or w['id']) + ".html"
        (outdir / name).write_text(html_doc, encoding="utf-8")
        print(f"✔ wrote {outdir/name}")

if __name__ == "__main__":
    main()
