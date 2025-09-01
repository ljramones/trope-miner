#!/usr/bin/env python3
# review/scripts/queue_cli.py
import argparse, os, sqlite3, shutil, textwrap, re, uuid, time, random

# --- display-only quote mapping (keeps string length stable) ---------------
DISPLAY_CHAR_MAP = str.maketrans({"Ò":"“","Ó":"”","Õ":"’","Ô":"—","Ê":"—"})
def display_fix_quotes(s: str) -> str:
    return s.translate(DISPLAY_CHAR_MAP) if s else s

# --- schema bootstrap (view for latest human decision) ---------------------
def ensure_review_schema(conn: sqlite3.Connection) -> None:
    conn.executescript("""
    PRAGMA foreign_keys=ON;
    CREATE TABLE IF NOT EXISTS trope_finding_human(
      id TEXT PRIMARY KEY,
      finding_id TEXT NOT NULL,
      decision TEXT NOT NULL CHECK(decision IN ('accept','reject','edit')),
      corrected_start INTEGER,
      corrected_end INTEGER,
      corrected_trope_id TEXT,
      note TEXT,
      reviewer TEXT,
      created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
      FOREIGN KEY(finding_id) REFERENCES trope_finding(id) ON DELETE CASCADE
    );
    CREATE INDEX IF NOT EXISTS idx_tfh_finding ON trope_finding_human(finding_id);

    CREATE VIEW IF NOT EXISTS v_latest_human AS
    SELECT h.*
    FROM trope_finding_human h
    JOIN (
      SELECT finding_id, MAX(created_at) AS mx
      FROM trope_finding_human
      GROUP BY finding_id
    ) last ON last.finding_id=h.finding_id AND last.mx=h.created_at;
    """)
    conn.commit()

# --- helpers ----------------------------------------------------------------
def colorize(s, on=True, style="hl"):
    if not on:
        return s
    # yellow background for span; dim gray for meta
    if style == "hl":  return "\x1b[30;43m" + s + "\x1b[0m"
    if style == "meta":return "\x1b[2m" + s + "\x1b[0m"
    if style == "title":return "\x1b[1m" + s + "\x1b[0m"
    return s

def term_width(default=100):
    try:
        return shutil.get_terminal_size((default, 24)).columns
    except Exception:
        return default

def clamp(a, lo, hi): return max(lo, min(hi, a))

def excerpt(scene_text: str, base: int, start_abs: int, end_abs: int,
            pre=160, post=160, color=True, width=None):
    """Return a single-line excerpt around [start_abs,end_abs) with ANSI highlight."""
    if width is None: width = term_width()
    t = scene_text
    s_rel = clamp(start_abs - base, 0, len(t))
    e_rel = clamp(end_abs   - base, s_rel, len(t))
    left_i  = max(0, s_rel - pre)
    right_i = min(len(t), e_rel + post)
    left  = t[left_i:s_rel]
    mid   = t[s_rel:e_rel]
    right = t[e_rel:right_i]
    # add ellipses if we sliced
    pre_ellipsis  = "…" if left_i > 0 else ""
    post_ellipsis = "…" if right_i < len(t) else ""
    out = pre_ellipsis + left + colorize(mid, color, "hl") + right + post_ellipsis
    # collapse newlines into spaces for a single-line display
    out = re.sub(r'\s+', ' ', out.strip())
    # soft wrap for terminal width
    wrapped = textwrap.fill(out, width=width)
    return wrapped

def parse_edit(expr: str, cur_start: int, cur_end: int):
    """
    Accept forms:
      - '123 145' or '123,145'          → absolute
      - '+10 -5'   or '+10,-5'          → relative (delta start, delta end)
    """
    m = re.match(r'^\s*([+-]?\d+)\s*[, ]\s*([+-]?\d+)\s*$', expr or '')
    if not m: return None
    a, b = m.group(1), m.group(2)
    if a.startswith(('+','-')) or b.startswith(('+','-')):
        return cur_start + int(a), cur_end + int(b)
    return int(a), int(b)

def count_remaining(conn, where_sql, args):
    row = conn.execute(f"""
      SELECT COUNT(1)
      FROM trope_finding f
      LEFT JOIN v_latest_human h ON h.finding_id=f.id
      WHERE {where_sql}
    """, args).fetchone()
    return int(row[0]) if row else 0

def build_filters(args):
    where = ["h.finding_id IS NULL"]
    vals  = []
    if args.work_id:
        where.append("f.work_id = ?"); vals.append(args.work_id)
    if args.trope_id:
        where.append("f.trope_id = ?"); vals.append(args.trope_id)
    if args.min_conf is not None:
        where.append("COALESCE(f.confidence,0) >= ?"); vals.append(args.min_conf)
    if args.max_conf is not None:
        where.append("COALESCE(f.confidence,0) <= ?"); vals.append(args.max_conf)
    return " AND ".join(where), vals

def order_sql(kind: str):
    k = (kind or "uncertain").lower()
    if k == "newest":  return "COALESCE(f.created_at,'0000') DESC"
    if k == "highest": return "COALESCE(f.confidence,0) DESC"
    if k == "random":  return "RANDOM()"
    return "ABS(COALESCE(f.confidence,0.5)-0.5) ASC, COALESCE(f.created_at,'0000') DESC"

def get_next(conn, where_sql, args, ord_sql):
    return conn.execute(f"""
      SELECT
        f.id, f.work_id, f.scene_id, f.trope_id, COALESCE(f.confidence,0.0) AS confidence,
        f.evidence_start AS start, f.evidence_end AS end, f.rationale,
        t.name AS trope,
        s.idx AS scene_idx, s.char_start, s.char_end,
        w.title, w.author, w.norm_text
      FROM trope_finding f
      JOIN scene s ON s.id=f.scene_id
      JOIN work  w ON w.id=f.work_id
      JOIN trope t ON t.id=f.trope_id
      LEFT JOIN v_latest_human h ON h.finding_id=f.id
      WHERE {where_sql}
      ORDER BY {ord_sql}
      LIMIT 1
    """, args).fetchone()

def insert_decision(conn, finding_id, decision, note=None, reviewer=None):
    conn.execute(
        "INSERT INTO trope_finding_human(id,finding_id,decision,note,reviewer) VALUES(?,?,?,?,?)",
        (str(uuid.uuid4()), finding_id, decision, note, reviewer)
    )
    conn.commit()

def apply_edit(conn, finding_id, start, end, trope_id=None, note=None, reviewer=None):
    # clamp to doc length and enforce start < end
    row = conn.execute("""
      SELECT f.work_id, COALESCE(length(w.norm_text),0) AS n
      FROM trope_finding f JOIN work w ON w.id=f.work_id
      WHERE f.id=?""", (finding_id,)).fetchone()
    if not row:
        return False, "finding not found"
    N = int(row["n"])
    start = clamp(start, 0, N)
    end   = clamp(end,   0, N)
    if end <= start:
        return False, "end must be > start"

    # history row
    conn.execute("""
      INSERT INTO trope_finding_human
        (id,finding_id,decision,corrected_start,corrected_end,corrected_trope_id,note,reviewer)
      VALUES (?,?,?,?,?,?,?,?)""",
      (str(uuid.uuid4()), finding_id, "edit", start, end, trope_id, note, reviewer)
    )
    # update canonical finding
    if trope_id:
        conn.execute("UPDATE trope_finding SET evidence_start=?, evidence_end=?, trope_id=? WHERE id=?",
                     (start, end, trope_id, finding_id))
    else:
        conn.execute("UPDATE trope_finding SET evidence_start=?, evidence_end=? WHERE id=?",
                     (start, end, finding_id))
    conn.commit()
    return True, None

def main():
    ap = argparse.ArgumentParser(description="Terminal review queue for trope findings (Accept/Reject/Edit/Next).")
    ap.add_argument("--db", required=True, help="SQLite DB path (e.g., ../ingester/tropes.db)")
    ap.add_argument("--work-id")
    ap.add_argument("--trope-id")
    ap.add_argument("--min-conf", type=float)
    ap.add_argument("--max-conf", type=float)
    ap.add_argument("--order", default="uncertain", choices=["uncertain","newest","highest","random"])
    ap.add_argument("--limit", type=int, default=0, help="stop after N decisions (0 = unlimited)")
    ap.add_argument("--full", action="store_true", help="print full scene instead of excerpt")
    ap.add_argument("--no-color", action="store_true")
    ap.add_argument("--reviewer", default=os.getenv("REVIEWER",""))
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    ensure_review_schema(conn)

    where_sql, vals = build_filters(args)
    ord_sql = order_sql(args.order)

    total = count_remaining(conn, where_sql, vals)
    decided = 0
    print(colorize(f"Queue: {total} unreviewed finding(s) — order={args.order}", True, "title"))

    while True:
        row = get_next(conn, where_sql, vals, ord_sql)
        if not row:
            print("No more unreviewed findings matching current filters.")
            break

        # scene slice + display
        s0, s1 = int(row["char_start"]), int(row["char_end"])
        scene_text = display_fix_quotes((row["norm_text"] or "")[s0:s1])
        title = f"{row['title']} — Scene #{row['scene_idx']}"
        meta  = f"work={row['work_id']}  trope={row['trope']}  conf={row['confidence']:.2f}  span=[{row['start']}–{row['end']}]"

        print()
        print(colorize(title, True, "title"))
        print(colorize(meta, True, "meta"))

        if args.full:
            # full scene with a single highlight block (best-effort, may wrap)
            left = scene_text[:max(0, row["start"]-s0)]
            mid  = scene_text[max(0, row["start"]-s0):max(0, row["end"]-s0)]
            right= scene_text[max(0, row["end"]-s0):]
            print(left + colorize(mid, not args.no_color, "hl") + right)
        else:
            print(excerpt(scene_text, s0, int(row["start"]), int(row["end"]),
                          pre=200, post=200, color=not args.no_color))

        # prompt
        print(colorize("[A]ccept  [R]eject  [E]dit  [N]ext  [Q]uit", True, "meta"))
        choice = input("> ").strip().lower()

        if choice in ("q","quit","x","exit"):
            break
        elif choice in ("n","next","s","skip",""):
            # skip; exclude this id by adding a one-off "after" filter (cheap way: insert temp row into h to skip? Better: just change WHERE)
            # We won't mutate WHERE; we'll just insert a transient reject? No. We keep it simple: add a small in-memory set.
            # Easiest: insert a temporary no-op decision? Avoid polluting history. So just reshuffle order when random, or continue (we may get same item).
            # To guarantee forward progress, temporarily consider it "reviewed" in this loop by adding a local exclusion.
            vals2 = vals + [row["id"]]
            row2 = conn.execute(f"""
              SELECT COUNT(1) FROM trope_finding f LEFT JOIN v_latest_human h ON h.finding_id=f.id
              WHERE {where_sql} AND f.id <> ?""", vals2).fetchone()
            if int(row2[0]) == 0:
                print("No more items after skipping.")
                break
            # mutate WHERE for this process only
            where_sql += " AND f.id <> ?"
            vals.append(row["id"])
            continue

        elif choice in ("a","accept","y","yes"):
            insert_decision(conn, row["id"], "accept", reviewer=args.reviewer or None)
            decided += 1

        elif choice in ("r","reject","no"):
            insert_decision(conn, row["id"], "reject", reviewer=args.reviewer or None)
            decided += 1

        elif choice in ("e","edit"):
            print("Edit span. Enter new absolute 'start,end' (e.g. 10366,10473) or relative '+10,-5'.")
            inp = input("start,end: ").strip()
            se = parse_edit(inp, int(row["start"]), int(row["end"]))
            if not se:
                print("! Could not parse. Skipping edit.")
                continue
            ok, err = apply_edit(conn, row["id"], se[0], se[1], note="cli-edit", reviewer=args.reviewer or None)
            if not ok:
                print(f"! Edit failed: {err}")
                continue
            # optional fast accept after edit
            ch = input("Accept after edit? [y/N]: ").strip().lower()
            if ch in ("y","yes"):
                insert_decision(conn, row["id"], "accept", reviewer=args.reviewer or None)
                decided += 1
        else:
            print("…unknown command; use A/R/E/N/Q")
            continue

        # progress / stopping
        remaining = count_remaining(conn, where_sql, vals)
        print(colorize(f"✓ saved. Remaining: {remaining}", True, "meta"))
        if args.limit and decided >= args.limit:
            print(f"Decision limit reached ({args.limit}).")
            break

    print(f"Done. Decisions recorded: {decided}")

if __name__ == "__main__":
    main()
