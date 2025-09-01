# scripts/calibrate_mini.py
import argparse, sqlite3, random, math
from collections import defaultdict

def iou(a,b):
    # a,b = (start,end)
    inter = max(0, min(a[1],b[1]) - max(a[0],b[0]))
    union = max(a[1]-a[0],0) + max(b[1]-b[0],0) - inter
    return 0.0 if union<=0 else inter/union

def fetch(conn, sql, args=()):
    cur = conn.execute(sql, args); cols=[c[0] for c in cur.description]
    return [dict(zip(cols,row)) for row in cur.fetchall()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--db', required=True)
    ap.add_argument('--k', type=int, default=10, help="random sample size if scene-ids not provided")
    ap.add_argument('--scene-ids', help="comma-separated scene ids")
    ap.add_argument('--threshold', type=float, default=0.5)
    ap.add_argument('--iou', type=float, default=0.3)
    args = ap.parse_args()

    DB = sqlite3.connect(args.db); DB.row_factory = sqlite3.Row

    if args.scene_ids:
        scene_ids = [s.strip() for s in args.scene_ids.split(',') if s.strip()]
    else:
        # choose k recent scenes that have at least one finding
        rows = fetch(DB, """
          SELECT DISTINCT s.id
          FROM scene s JOIN trope_finding f ON f.scene_id = s.id
          ORDER BY s.id DESC
        """)
        pool = [r['id'] for r in rows]
        scene_ids = random.sample(pool, min(args.k, len(pool)))

    total_tp=total_fp=total_fn=0
    per_trope = defaultdict(lambda: {'tp':0,'fp':0,'fn':0})

    for sid in scene_ids:
        # predictions at threshold τ
        preds = fetch(DB, """
          SELECT id, trope_id, evidence_start AS s, evidence_end AS e, confidence
          FROM trope_finding
          WHERE scene_id=? AND confidence >= ?
        """, (sid, args.threshold))

        # ground truth = accepted latest human decisions (edits allowed)
        gts = fetch(DB, """
          SELECT f.id as fid, COALESCE(h.corrected_trope_id, f.trope_id) AS trope_id,
                 COALESCE(h.corrected_start, f.evidence_start) AS s,
                 COALESCE(h.corrected_end,   f.evidence_end)   AS e
          FROM trope_finding f
          JOIN v_latest_human h ON h.finding_id = f.id AND h.decision='accept'
          WHERE f.scene_id = ?
        """, (sid,))

        # match by trope_id + IoU≥θ
        matched_pred=set(); matched_gt=set()
        for gi, gt in enumerate(gts):
            best=None; best_iou=0.0; best_pi=None
            for pi, pr in enumerate(preds):
                if pr['trope_id'] != gt['trope_id'] or pi in matched_pred: continue
                j = iou((pr['s'],pr['e']), (gt['s'],gt['e']))
                if j>best_iou:
                    best_iou=j; best=pr; best_pi=pi
            if best and best_iou >= args.iou:
                matched_pred.add(best_pi); matched_gt.add(gi)
                total_tp += 1; per_trope[gt['trope_id']]['tp'] += 1

        # leftovers: FPs (unmatched preds) and FNs (unmatched gts)
        fps = [p for i,p in enumerate(preds) if i not in matched_pred]
        fns = [g for i,g in enumerate(gts)   if i not in matched_gt ]
        total_fp += len(fps); total_fn += len(fns)
        for p in fps: per_trope[p['trope_id']]['fp'] += 1
        for g in fns: per_trope[g['trope_id']]['fn'] += 1

    prec = total_tp / (total_tp + total_fp) if (total_tp+total_fp)>0 else 0.0
    rec  = total_tp / (total_tp + total_fn) if (total_tp+total_fn)>0 else 0.0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0

    print(f"Scenes: {len(scene_ids)}  τ={args.threshold:.2f}  IoU≥{args.iou:.2f}")
    print(f"TP={total_tp} FP={total_fp} FN={total_fn}")
    print(f"Precision={prec:.3f}  Recall={rec:.3f}  F1={f1:.3f}")
    print("\nPer-trope:")
    for t, m in sorted(per_trope.items(), key=lambda kv: kv[0]):
        tp,fp,fn = m['tp'],m['fp'],m['fn']
        p = tp/(tp+fp) if (tp+fp)>0 else 0.0
        r = tp/(tp+fn) if (tp+fn)>0 else 0.0
        f = 2*p*r/(p+r) if (p+r)>0 else 0.0
        print(f"  {t:20s}  TP={tp:3d} FP={fp:3d} FN={fn:3d}  P={p:.3f} R={r:.3f} F1={f:.3f}")

if __name__ == "__main__":
    main()
