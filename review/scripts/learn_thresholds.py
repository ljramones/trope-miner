# review/scripts/learn_thresholds.py
import argparse, sqlite3, json, time
from collections import defaultdict


def fetch(conn, sql, args=()):
    cur = conn.execute(sql, args);
    cols = [c[0] for c in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def best_threshold(points, method="f1"):
    """
    points: list of (score in [0,1], label) where label=1 for accept, 0 for reject.
    We scan unique scores as thresholds (>= t is positive) and pick t maximizing F1.
    Tie-breaker: smallest t achieving best F1.
    """
    if not points: return 0.5, dict(tp=0, fp=0, fn=0, prec=0.0, rec=0.0, f1=0.0)
    # uniq sorted descending for classic sweep
    uniq = sorted(set(s for s, _ in points), reverse=True)
    # Precompute counts
    total_pos = sum(1 for _, y in points if y == 1)
    total_neg = len(points) - total_pos
    best = (-1.0, 1.0, 0.0, dict(tp=0, fp=0, fn=total_pos, prec=0.0, rec=0.0, f1=0.0))  # (F1, t, P, stats)
    for t in uniq:
        tp = sum(1 for s, y in points if s >= t and y == 1)
        fp = sum(1 for s, y in points if s >= t and y == 0)
        fn = total_pos - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        key = (f1, -t)  # prefer lower t on tie
        if key > (best[0], -best[1]):
            best = (f1, t, prec, dict(tp=tp, fp=fp, fn=fn, prec=prec, rec=rec, f1=f1))
    return best[1], best[3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--db', required=True)
    ap.add_argument('--min-count', type=int, default=6, help='min # labeled (accept+reject) per trope')
    ap.add_argument('--out', help='optional thresholds.json output path')
    ap.add_argument('--write-table', action='store_true', help='write results to table trope_threshold')
    args = ap.parse_args()

    conn = sqlite3.connect(args.db);
    conn.row_factory = sqlite3.Row

    # labels: use latest decision; if edited, use corrected_trope_id to credit the right trope
    rows = fetch(conn, """
                       SELECT COALESCE(h.corrected_trope_id, f.trope_id)                              AS trope_id,
                              COALESCE(f.confidence, 0.0)                                             AS score,
                              CASE h.decision WHEN 'accept' THEN 1 WHEN 'reject' THEN 0 ELSE NULL END AS label
                       FROM trope_finding f
                                JOIN v_latest_human h ON h.finding_id = f.id
                       WHERE h.decision IN ('accept', 'reject')
                       """)

    by_trope = defaultdict(list)
    for r in rows:
        if r["label"] is None: continue
        by_trope[r["trope_id"]].append((float(r["score"]), int(r["label"])))

    results = {}
    for tid, pts in by_trope.items():
        if len(pts) < args.min_count: continue
        thr, stats = best_threshold(pts)
        results[tid] = dict(threshold=thr, **stats, n=len(pts))

    # Optionally write a table
    if args.write_table:
        conn.executescript("""
                           CREATE TABLE IF NOT EXISTS trope_threshold
                           (
                               trope_id
                               TEXT
                               PRIMARY
                               KEY,
                               threshold
                               REAL
                               NOT
                               NULL,
                               n
                               INTEGER
                               NOT
                               NULL,
                               tp
                               INTEGER
                               NOT
                               NULL,
                               fp
                               INTEGER
                               NOT
                               NULL,
                               fn
                               INTEGER
                               NOT
                               NULL,
                               prec
                               REAL
                               NOT
                               NULL,
                               rec
                               REAL
                               NOT
                               NULL,
                               f1
                               REAL
                               NOT
                               NULL,
                               method
                               TEXT
                               NOT
                               NULL
                               DEFAULT
                               'grid-f1',
                               updated_at
                               TEXT
                               NOT
                               NULL
                               DEFAULT (
                               strftime
                           (
                               '%Y-%m-%dT%H:%M:%SZ',
                               'now'
                           ))
                               );
                           """)
        for tid, r in results.items():
            conn.execute("""
                         INSERT INTO trope_threshold(trope_id, threshold, n, tp, fp, fn, prec, rec, f1, method, updated_at)
                         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'grid-f1',
                                 strftime('%Y-%m-%dT%H:%M:%SZ', 'now')) ON CONFLICT(trope_id) DO
                         UPDATE SET
                             threshold=excluded.threshold, n=excluded.n, tp=excluded.tp, fp=excluded.fp, fn=excluded.fn,
                             prec=excluded.prec, rec=excluded.rec, f1=excluded.f1, method =excluded.method, updated_at=excluded.updated_at
                         """, (tid, r["threshold"], r["n"], r["tp"], r["fp"], r["fn"], r["prec"], r["rec"], r["f1"]))
        conn.commit()

    # Optional JSON
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"✔ wrote {args.out}")

    # Console report
    if not results:
        print("No tropes met min-count; try lowering --min-count.")
        return

    print("trope_id,threshold,n,tp,fp,fn,prec,rec,f1")
    for tid, r in sorted(results.items()):
        print(
            f"{tid},{r['threshold']:.2f},{r['n']},{r['tp']},{r['fp']},{r['fn']},{r['prec']:.3f},{r['rec']:.3f},{r['f1']:.3f}")


if __name__ == "__main__":
    main()
