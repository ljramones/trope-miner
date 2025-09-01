#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calibration helper
------------------
Uses human decisions (accept/reject) to:
  - build a reliability curve (confidence bins vs empirical acceptance),
  - sweep thresholds → precision/recall/F1/accuracy, recommend one,
  - write CSV and optional PNG plot.

Examples:
  python scripts/calibrate_threshold.py --db ingester/tropes.db \
    --out out/calibration.csv --plot out/calibration.png

  # restrict to one work
  python scripts/calibrate_threshold.py --db ingester/tropes.db \
    --work-id <WORK_UUID> --out out/calib_one_work.csv --plot out/calib_one_work.png
"""
from __future__ import annotations
import argparse, os, csv, math, sqlite3, statistics as stats

try:
    import matplotlib.pyplot as plt  # optional (only when --plot)
except Exception:
    plt = None

def fetch_labeled(conn: sqlite3.Connection, work_id: str|None):
    """
    Return list of (confidence, label, finding_id). label=1 if accepted, 0 if rejected.
    Only rows with a human accept/reject decision are included.
    """
    base = """
    SELECT f.id, f.confidence, h.decision
    FROM trope_finding f
    JOIN v_latest_human h ON h.finding_id=f.id
    WHERE h.decision IN ('accept','reject')
    """
    params = ()
    if work_id:
        base += " AND f.work_id=?"
        params = (work_id,)
    rows = conn.execute(base, params).fetchall()
    out = []
    for fid, conf, dec in rows:
        if conf is None: continue
        y = 1 if (dec == 'accept') else 0
        out.append((float(conf), y, fid))
    return out

def sweep_thresholds(points, step=0.01):
    """
    points: list of (conf, y)
    returns: list of dicts per threshold
    """
    if not points:
        return []
    points = [(float(c), int(y)) for (c,y,_) in points]
    N = len(points)
    res = []
    t = 0.0
    while t <= 1.000001:
        tp=fp=tn=fn=0
        pred = [(1 if c>=t else 0, y) for (c,y) in points]
        for p,y in pred:
            if p==1 and y==1: tp+=1
            elif p==1 and y==0: fp+=1
            elif p==0 and y==0: tn+=1
            elif p==0 and y==1: fn+=1
        prec = tp/(tp+fp) if (tp+fp)>0 else 0.0
        rec  = tp/(tp+fn) if (tp+fn)>0 else 0.0
        f1   = (2*prec*rec)/(prec+rec) if (prec+rec)>0 else 0.0
        acc  = (tp+tn)/N if N>0 else 0.0
        res.append({
            "threshold": round(t,4), "n": N,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": round(prec,6), "recall": round(rec,6),
            "f1": round(f1,6), "accuracy": round(acc,6),
            "pred_pos": tp+fp, "pred_neg": tn+fn,
            "pos_rate": (tp+fn)/N if N>0 else 0.0
        })
        t += step
    return res

def reliability_bins(points, bins=10):
    """
    Equal-width bins in [0,1]. Returns list of dicts:
      {"bin_lo","bin_hi","count","mean_conf","emp_accept"}
    """
    out = []
    if not points: return out
    width = 1.0/bins
    for i in range(bins):
        lo = i*width
        hi = (i+1)*width if i<bins-1 else 1.000001
        pts = [(c,y) for (c,y,_) in points if (c>=lo and c<hi if i<bins-1 else c>=lo and c<=1.0)]
        n = len(pts)
        if n==0:
            out.append({"bin_lo":lo, "bin_hi":hi, "count":0, "mean_conf":0.0, "emp_accept":0.0})
            continue
        mean_conf = sum(c for c,_ in pts)/n
        emp_acc   = sum(y for _,y in pts)/n
        out.append({"bin_lo":lo, "bin_hi":hi, "count":n,
                    "mean_conf":mean_conf, "emp_accept":emp_acc})
    return out

def expected_calibration_error(bins):
    # ECE = sum_k (n_k/N)*|acc_k - conf_k|
    N = sum(b["count"] for b in bins)
    if N==0: return 0.0
    ece = 0.0
    for b in bins:
        if b["count"]==0: continue
        ece += (b["count"]/N) * abs(b["emp_accept"] - b["mean_conf"])
    return ece

def choose_threshold(rows, objective="f1", min_precision=0.7, min_recall=0.0):
    """
    objective:
      - "f1": maximize F1
      - "f1@precision": maximize F1 subject to precision >= min_precision
      - "precision@recall": maximize precision subject to recall >= min_recall
    """
    best = None
    if not rows: return None
    if objective == "f1":
        key = lambda r: (r["f1"], r["precision"], -r["threshold"])  # tie-break: higher prec, lower threshold
        best = max(rows, key=key)
    elif objective == "f1@precision":
        cand = [r for r in rows if r["precision"] >= min_precision]
        if cand:
            best = max(cand, key=lambda r: (r["f1"], r["recall"], -r["threshold"]))
        else:
            best = max(rows, key=lambda r: (r["precision"], r["f1"], -r["threshold"]))
    elif objective == "precision@recall":
        cand = [r for r in rows if r["recall"] >= min_recall]
        if cand:
            best = max(cand, key=lambda r: (r["precision"], r["f1"], -r["threshold"]))
        else:
            best = max(rows, key=lambda r: (r["recall"], r["precision"], -r["threshold"]))
    return best

def write_csv(path, sweep_rows, rel_bins, meta):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["# meta"])
        for k,v in meta.items():
            w.writerow([k, v])
        w.writerow([])
        w.writerow(["threshold","n","tp","fp","tn","fn","precision","recall","f1","accuracy","pred_pos","pred_neg","pos_rate"])
        for r in sweep_rows:
            w.writerow([r[k] for k in ["threshold","n","tp","fp","tn","fn","precision","recall","f1","accuracy","pred_pos","pred_neg","pos_rate"]])
        w.writerow([])
        w.writerow(["# reliability_bins"])
        w.writerow(["bin_lo","bin_hi","count","mean_conf","emp_accept"])
        for b in rel_bins:
            w.writerow([round(b["bin_lo"],4), round(b["bin_hi"],4), b["count"], round(b["mean_conf"],6), round(b["emp_accept"],6)])

def make_plot(path, sweep_rows, rel_bins, best_row):
    if plt is None:
        print("[calib] matplotlib not available; skipping plot.")
        return
    import matplotlib.pyplot as plt
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    # Figure: 1) reliability, 2) metrics vs threshold
    fig = plt.figure(figsize=(9,4.8))
    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)

    # Reliability
    ax1.plot([0,1],[0,1], linestyle="--", linewidth=1)
    xs = [min(1,max(0,b["mean_conf"])) for b in rel_bins if b["count"]>0]
    ys = [min(1,max(0,b["emp_accept"])) for b in rel_bins if b["count"]>0]
    sizes = [max(10, 10*math.log(b["count"]+1, 1.5)) for b in rel_bins if b["count"]>0]
    ax1.scatter(xs, ys, s=sizes)
    ece = expected_calibration_error(rel_bins)
    ax1.set_title(f"Reliability (ECE={ece:.3f})")
    ax1.set_xlabel("Mean predicted confidence")
    ax1.set_ylabel("Empirical accept rate")

    # Metrics vs threshold
    ts = [r["threshold"] for r in sweep_rows]
    ax2.plot(ts, [r["precision"] for r in sweep_rows], label="precision")
    ax2.plot(ts, [r["recall"]    for r in sweep_rows], label="recall")
    ax2.plot(ts, [r["f1"]        for r in sweep_rows], label="f1")
    ax2.plot(ts, [r["accuracy"]  for r in sweep_rows], label="accuracy")
    if best_row:
        ax2.axvline(best_row["threshold"], linestyle="--")
        ax2.text(best_row["threshold"], 0.02, f"  t*={best_row['threshold']:.2f}", rotation=90, va="bottom")
    ax2.set_ylim(0,1)
    ax2.set_xlim(0,1)
    ax2.grid(alpha=0.2)
    ax2.legend()
    ax2.set_title("Metrics vs threshold")
    ax2.set_xlabel("threshold")

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    print(f"[calib] wrote plot → {path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--out", required=True, help="CSV path")
    ap.add_argument("--plot", help="PNG path (optional)")
    ap.add_argument("--work-id", help="restrict to a single work")
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--step", type=float, default=0.01)
    ap.add_argument("--objective", choices=["f1","f1@precision","precision@recall"], default="f1@precision")
    ap.add_argument("--min-precision", type=float, default=0.70)
    ap.add_argument("--min-recall", type=float, default=0.10)
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row

    pts = fetch_labeled(conn, args.work_id)
    if not pts:
        raise SystemExit("No labeled (accepted/rejected) findings found. Make some decisions in the Review UI first.")

    sweep = sweep_thresholds([(c,y) for (c,y,_) in pts], step=args.step)
    rel = reliability_bins(pts, bins=args.bins)
    best = choose_threshold(sweep, objective=args.objective, min_precision=args.min_precision, min_recall=args.min_recall)

    meta = {
        "n_labeled": len(pts),
        "objective": args.objective,
        "min_precision": args.min_precision,
        "min_recall": args.min_recall,
        "recommended_threshold": best["threshold"] if best else None,
        "ece": round(expected_calibration_error(rel), 6)
    }
    write_csv(args.out, sweep, rel, meta)
    print(f"[calib] wrote CSV → {args.out}")
    if best:
        print(f"[calib] recommended threshold (objective={args.objective}): {best['threshold']:.2f} "
              f"(precision={best['precision']:.2f}, recall={best['recall']:.2f}, f1={best['f1']:.2f})")
    if args.plot:
        make_plot(args.plot, sweep, rel, best)

if __name__ == "__main__":
    main()
