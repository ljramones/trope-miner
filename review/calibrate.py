#!/usr/bin/env python3
"""
Calibration helper:
- Uses latest human decision per finding (accept=1, reject=0).
- Plots reliability (confidence bin vs. accept-rate).
- Finds threshold that maximizes F1 on reviewed findings.
- Writes PNG + JSON summary.

Usage:
  python review/calibrate.py --db ingester/tropes.db --out-dir ingester/out
"""
from __future__ import annotations
import argparse, json, os, sqlite3
from pathlib import Path
import math
import matplotlib.pyplot as plt  # stdlib-friendly; no seaborn


def fetch(db: str):
    con = sqlite3.connect(db);
    con.row_factory = sqlite3.Row
    rows = con.execute("""
                       SELECT f.id, f.confidence, h.decision
                       FROM trope_finding f
                                JOIN (SELECT h1.finding_id, h1.decision
                                      FROM trope_finding_human h1
                                               JOIN (SELECT finding_id, MAX(created_at) mx
                                                     FROM trope_finding_human
                                                     GROUP BY finding_id) last
                                                    ON last.finding_id = h1.finding_id AND last.mx = h1.created_at) h
                                     ON h.finding_id = f.id
                       WHERE h.decision IN ('accept', 'reject')
                         AND f.confidence IS NOT NULL
                       """).fetchall()
    con.close()
    return rows


def reliability(rows, bins=10):
    # bin by confidence into [0,1]
    buckets = [[] for _ in range(bins)]
    for r in rows:
        c = max(0.0, min(1.0, float(r["confidence"])))
        b = min(bins - 1, int(c * bins))
        buckets[b].append(1 if r["decision"] == "accept" else 0)
    xs, ys, ns = [], [], []
    for i, b in enumerate(buckets):
        if not b: continue
        xs.append((i + 0.5) / bins)
        ys.append(sum(b) / len(b))
        ns.append(len(b))
    return xs, ys, ns


def precision_recall(rows, threshold):
    preds = [(float(r["confidence"]) >= threshold) for r in rows]
    labels = [1 if r["decision"] == "accept" else 0 for r in rows]
    tp = sum(1 for p, l in zip(preds, labels) if p and l)
    fp = sum(1 for p, l in zip(preds, labels) if p and not l)
    fn = sum(1 for p, l in zip(preds, labels) if (not p) and l)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1, tp, fp, fn


def pick_threshold(rows):
    best = (0.0, 0.0, 0.0, 0.0)  # f1, thr, prec, rec
    for k in range(0, 101):
        thr = k / 100.0
        prec, rec, f1, *_ = precision_recall(rows, thr)
        if f1 > best[0]: best = (f1, thr, prec, rec)
    return {"threshold": best[1], "f1": best[0], "precision": best[2], "recall": best[3]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="ingester/tropes.db")
    ap.add_argument("--out-dir", default="ingester/out")
    args = ap.parse_args()

    rows = fetch(args.db)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    if not rows:
        print("[calib] no reviewed findings; review a few first")
        return

    xs, ys, ns = reliability(rows)
    # Plot reliability
    plt.figure()
    plt.plot(xs, ys, marker='o', label='empirical')
    plt.plot([0, 1], [0, 1], linestyle='--', label='ideal')
    plt.xlabel('model confidence')
    plt.ylabel('observed accept rate')
    plt.title('Reliability (reviewed findings)')
    plt.legend()
    png = Path(args.out_dir) / "calibration.png"
    plt.savefig(png, bbox_inches='tight')
    print(f"[calib] wrote {png.resolve()}")

    picked = pick_threshold(rows)
    summary = {
        "n_reviewed": len(rows),
        "recommended_threshold": picked["threshold"],
        "max_f1": picked["f1"],
        "precision_at_thr": picked["precision"],
        "recall_at_thr": picked["recall"],
        "bins": [{"x": x, "y": y, "n": int(n)} for x, y, n in zip(xs, ys, ns)]
    }
    js = Path(args.out_dir) / "calibration.json"
    js.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[calib] wrote {js.resolve()}")
    print(
        f"[calib] recommend --threshold ~ {picked['threshold']:.2f} (F1={picked['f1']:.2f}, P={picked['precision']:.2f}, R={picked['recall']:.2f})")


if __name__ == "__main__":
    main()
