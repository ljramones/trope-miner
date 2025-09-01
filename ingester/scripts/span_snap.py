#!/usr/bin/env python3
import argparse, sqlite3, re, sys

BOUND = re.compile(r'[.!?]"?\s')

def snap_within(text: str, start: int, end: int, max_len: int = 280):
    n = len(text)
    start = max(0, min(start, n)); end = max(0, min(end, n))
    if end < start: start, end = end, start

    # expand to nearest boundary tokens
    left = 0
    for m in BOUND.finditer(text[:start]):
        left = m.end()
    right = n
    m = BOUND.search(text, end)
    if m: right = m.end()

    # cap length by centering on original span
    span_len = min(max_len, n)
    if right - left > span_len:
        center = (start + end) // 2
        half = span_len // 2
        left = max(0, min(center - half, n - span_len))
        right = left + span_len

    return left, right

def main():
    ap = argparse.ArgumentParser(description="Snap finding evidence spans to sentence-ish boundaries.")
    ap.add_argument("--db", default="ingester/tropes.db")
    ap.add_argument("--work-id", default=None, help="Limit to a single work (optional)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--max-len", type=int, default=280)
    args = ap.parse_args()
    if not (args.dry_run or args.apply):
        print("Specify --dry-run or --apply", file=sys.stderr); sys.exit(2)

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row

    work_clause = ""
    params = ()
    if args.work_id:
        work_clause = "WHERE f.work_id=?"
        params = (args.work_id,)

    q = conn.execute(f"""
      SELECT f.id, f.scene_id, f.evidence_start, f.evidence_end,
             s.char_start AS s_start, s.char_end AS s_end, w.norm_text
      FROM trope_finding f
      JOIN scene s ON s.id = f.scene_id
      JOIN work  w ON w.id = f.work_id
      {work_clause}
    """, params)

    updates = []
    for r in q:
        sid, sstart, send = r["scene_id"], int(r["s_start"]), int(r["s_end"])
        txt = r["norm_text"][sstart:send]
        es, ee = int(r["evidence_start"]), int(r["evidence_end"])
        # convert to scene-relative
        rel_s, rel_e = max(0, es - sstart), max(0, ee - sstart)
        new_s_rel, new_e_rel = snap_within(txt, rel_s, rel_e, max_len=args.max_len)
        new_s_abs, new_e_abs = sstart + new_s_rel, sstart + new_e_rel

        # only change if different and still within bounds
        if (new_s_abs, new_e_abs) != (es, ee) and sstart <= new_s_abs < new_e_abs <= send:
            updates.append((r["id"], es, ee, new_s_abs, new_e_abs))

    if args.dry_run:
        for fid, old_s, old_e, ns, ne in updates[:50]:
            print(f"{fid[:8]} {old_s}-{old_e}  ->  {ns}-{ne}")
        print(f"(previewed {min(50,len(updates))} of {len(updates)} updates)")
    if args.apply and updates:
        with conn:
            for fid, old_s, old_e, ns, ne in updates:
                conn.execute("UPDATE trope_finding SET evidence_start=?, evidence_end=? WHERE id=?",
                             (ns, ne, fid))
        print(f"Applied {len(updates)} span updates.")

if __name__ == "__main__":
    main()
