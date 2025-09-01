#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stamp findings with calibration_version + threshold_used.

Examples:
  python scripts/stamp_calibration.py --db ingester/tropes.db \
    --work-id <WORK_UUID> --version calib-2025-08-31 --threshold 0.55

If --since is provided, only rows with created_at >= since are updated.
"""
from __future__ import annotations
import argparse, sqlite3

def ensure_cols(conn):
    cur = conn.cursor()
    cols = {r[1] for r in cur.execute("PRAGMA table_info(trope_finding)")}
    if "calibration_version" not in cols:
        cur.execute("ALTER TABLE trope_finding ADD COLUMN calibration_version TEXT;")
    if "threshold_used" not in cols:
        cur.execute("ALTER TABLE trope_finding ADD COLUMN threshold_used REAL;")
    conn.commit()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--work-id", help="restrict stamping to this work")
    ap.add_argument("--version", required=True)
    ap.add_argument("--threshold", type=float, required=True)
    ap.add_argument("--since", help="ISO timestamp filter (optional)")
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    ensure_cols(conn)

    q = "UPDATE trope_finding SET calibration_version=?, threshold_used=? WHERE 1=1"
    params = [args.version, args.threshold]
    if args.work_id:
        q += " AND work_id=?"
        params.append(args.work_id)
    if args.since:
        q += " AND created_at>=?"
        params.append(args.since)
    q += " AND (calibration_version IS NULL OR calibration_version='')"

    cur = conn.cursor()
    cur.execute(q, tuple(params))
    conn.commit()
    print(f"[stamp] updated rows: {cur.rowcount}")

if __name__ == "__main__":
    main()
