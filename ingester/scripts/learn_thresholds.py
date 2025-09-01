#!/usr/bin/env python3
"""
learn_thresholds.py — Per-trope threshold learning from human labels.

Data:
- Uses v_latest_human (accept/reject) joined to trope_finding (adj_conf)
- Reconstructs raw_conf via adj_conf / prior where prior = trope_sanity.weight
  (fallback prior=1.0 if missing)
- Features per sample: [raw_conf, prior, raw_conf*prior]
- Label: accept=1, reject=0

Output:
- Table trope_thresholds(trope_id PRIMARY KEY, threshold REAL, samples INT, pos INT, neg INT, objective TEXT, updated_at TEXT)
- Optional CSV with learned rows.

Usage:
  python ingester/scripts/learn_thresholds.py \
    --db ingester/tropes.db \
    [--work-id <UUID>] \
    [--min-samples 8] \
    [--objective f1|f1@precision|precision@recall] \
    [--min-precision 0.75] \
    [--min-recall 0.10] \
    [--step 0.01] \
    [--out out/trope_thresholds.csv]
"""
from __future__ import annotations
import argparse, math, sqlite3, csv, sys
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

def sigmoid(z: float) -> float:
    if z >= 0:
        ez = math.exp(-z); return 1.0 / (1.0 + ez)
    ez = math.exp(z); return ez / (1.0 + ez)

def fit_logistic(X: List[List[float]], y: List[int], l2: float = 1e-3, lr: float = 0.1, iters: int = 400) -> List[float]:
    """Minimal batch logistic with L2. X rows are [1, raw, prior, raw*prior]."""
    if not X: return [0.0, 1.0, 0.0, 0.0]
    n, d = len(X), len(X[0])
    w = [0.0]*d
    for _ in range(iters):
        # gradient
        grad = [0.0]*d
        for i in range(n):
            z = sum(w[j]*X[i][j] for j in range(d))
            p = sigmoid(z)
            diff = (p - y[i])
            for j in range(d):
                grad[j] += diff*X[i][j]
        for j in range(d):
            grad[j] = grad[j]/n + l2*w[j]
            w[j] -= lr*grad[j]
    return w

def predict_prob(w: List[float], x: List[float]) -> float:
    return sigmoid(sum(w[j]*x[j] for j in range(len(w))))

def f1(p:int, r:int, tp:int) -> float:
    if p+r == 0: return 0.0
    prec = tp/p if p>0 else 0.0
    rec  = tp/r if r>0 else 0.0
    if prec+rec == 0: return 0.0
    return 2*prec*rec/(prec+rec)

def sweep_threshold(adj: List[float], y: List[int], step: float,
                    objective: str, min_prec: float, min_rec: float) -> float:
    """Return best threshold on adjusted confidence."""
    best_t, best_score = 0.5, -1.0
    # Candidate thresholds = unique adj values + grid (for robustness)
    vals = sorted(set([round(v,3) for v in adj] + [i*step for i in range(int(1/step)+1)]))
    P = sum(y); N = len(y)-P
    for t in vals:
        tp = sum(1 for yi, ai in zip(y, adj) if yi==1 and ai>=t)
        fp = sum(1 for yi, ai in zip(y, adj) if yi==0 and ai>=t)
        fn = sum(1 for yi, ai in zip(y, adj) if yi==1 and ai< t)
        prec = tp/(tp+fp) if (tp+fp)>0 else 0.0
        rec  = tp/(tp+fn) if (tp+fn)>0 else 0.0
        if objective == "f1":
            score = f1(tp+fp, tp+fn, tp)
        elif objective == "f1@precision":
            if prec < min_prec: continue
            score = f1(tp+fp, tp+fn, tp)
        elif objective == "precision@recall":
            if rec < min_rec: continue
            score = prec
        else:
            score = f1(tp+fp, tp+fn, tp)
        if score > best_score:
            best_score, best_t = score, t
    return best_t

def ensure_table(conn: sqlite3.Connection) -> None:
    conn.execute("""
    CREATE TABLE IF NOT EXISTS trope_thresholds(
      trope_id   TEXT PRIMARY KEY,
      threshold  REAL NOT NULL,
      samples    INTEGER NOT NULL,
      pos        INTEGER NOT NULL,
      neg        INTEGER NOT NULL,
      objective  TEXT,
      updated_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
    );
    """)
    conn.commit()

def fetch_samples(conn: sqlite3.Connection, work_id: Optional[str]) -> Dict[str, List[Tuple[float,float,float,int]]]:
    """
    Returns per-trope samples as list of tuples:
      (raw_conf, prior, adj_conf, label) ; label: 1=accept, 0=reject
    """
    where = "WHERE v.decision IN ('accept','reject')"
    params: Tuple = ()
    if work_id:
        where += " AND f.work_id = ?"
        params = (work_id,)

    q = f"""
    SELECT f.trope_id,
           COALESCE(f.confidence,0.0) AS adj_conf,
           COALESCE(ts.weight,1.0)     AS prior,
           v.decision
    FROM trope_finding f
    JOIN v_latest_human v ON v.finding_id = f.id
    LEFT JOIN trope_sanity ts ON ts.scene_id = f.scene_id AND ts.trope_id = f.trope_id
    {where}
    """
    out: Dict[str, List[Tuple[float,float,float,int]]] = defaultdict(list)
    for row in conn.execute(q, params).fetchall():
        adj = float(row[1] or 0.0)
        pr  = float(row[2] or 1.0)
        pr  = pr if pr>0 else 1.0
        raw = max(0.0, min(1.0, adj / pr))
        y   = 1 if (row[3] == "accept") else 0
        out[row[0]].append((raw, pr, adj, y))
    return out

def main():
    ap = argparse.ArgumentParser(description="Learn per-trope thresholds from review labels.")
    ap.add_argument("--db", required=True)
    ap.add_argument("--work-id", help="Optional filter to a single work ID")
    ap.add_argument("--min-samples", type=int, default=8, help="Minimum labeled samples per trope")
    ap.add_argument("--objective", choices=["f1","f1@precision","precision@recall"], default="f1@precision")
    ap.add_argument("--min-precision", type=float, default=0.75)
    ap.add_argument("--min-recall", type=float, default=0.10)
    ap.add_argument("--step", type=float, default=0.01)
    ap.add_argument("--out", help="Optional CSV output of learned thresholds")
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    ensure_table(conn)

    per = fetch_samples(conn, args.work_id)
    rows_out = []
    for trope_id, samples in per.items():
        if len(samples) < args.min_samples:
            continue
        raw = [s[0] for s in samples]
        pr  = [s[1] for s in samples]
        adj = [s[2] for s in samples]
        y   = [s[3] for s in samples]

        # train tiny logistic (not persisted, just informs stability)
        X = [[1.0, r, p, r*p] for r,p in zip(raw, pr)]
        _w = fit_logistic(X, y)

        # choose a simple threshold on adjusted confidence, per objective
        th = sweep_threshold(adj, y, args.step, args.objective, args.min_precision, args.min_recall)

        pos = sum(y); neg = len(y)-pos
        conn.execute("""
            INSERT INTO trope_thresholds(trope_id, threshold, samples, pos, neg, objective, updated_at)
            VALUES(?,?,?,?,?,?,strftime('%Y-%m-%dT%H:%M:%fZ','now'))
            ON CONFLICT(trope_id) DO UPDATE SET
              threshold=excluded.threshold,
              samples=excluded.samples,
              pos=excluded.pos,
              neg=excluded.neg,
              objective=excluded.objective,
              updated_at=excluded.updated_at
        """, (trope_id, float(th), int(len(y)), int(pos), int(neg), args.objective))
        rows_out.append({"trope_id": trope_id, "threshold": th, "samples": len(y), "pos": pos, "neg": neg})

    conn.commit()

    if args.out:
        with open(args.out, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["trope_id","threshold","samples","pos","neg"])
            w.writeheader()
            for r in rows_out:
                w.writerow(r)

    print(f"[learn] wrote {len(rows_out)} trope thresholds.")
    conn.close()

if __name__ == "__main__":
    main()
