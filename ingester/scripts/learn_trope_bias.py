# scripts/learn_trope_bias.py
import sqlite3, math, sys
db = sys.argv[1] if len(sys.argv) > 1 else "ingester/tropes.db"
conn = sqlite3.connect(db)

# Build a simple accept rate with Laplace smoothing
rows = conn.execute("""
WITH agg AS (
  SELECT trope_id,
         SUM(CASE WHEN decision='accept' THEN 1 ELSE 0 END) AS acc,
         SUM(CASE WHEN decision='reject' THEN 1 ELSE 0 END) AS rej
  FROM trope_finding_human
  GROUP BY trope_id
)
SELECT trope_id, acc, rej, (acc+1.0)/(acc+rej+2.0) AS accept_rate
FROM agg
""").fetchall()

conn.execute("CREATE TABLE IF NOT EXISTS trope_bias (trope_id TEXT PRIMARY KEY, bias REAL)")
with conn:
    for tid, acc, rej, rate in rows:
        # map accept_rate ∈ (0,1) to a gentle multiplier ∈ [0.8, 1.2]
        bias = 0.8 + 0.4*rate
        conn.execute("INSERT INTO trope_bias(trope_id, bias) VALUES(?,?) ON CONFLICT(trope_id) DO UPDATE SET bias=excluded.bias", (tid, bias))
print("learned biases:", len(rows))
