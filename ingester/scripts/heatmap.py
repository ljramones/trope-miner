#!/usr/bin/env python3
"""
Chapter/Scene Heatmap
---------------------
Builds a scene-by-trope matrix (rows=scenes, cols=top-N tropes by frequency),
where each cell is the MAX confidence of that trope in that scene.

Outputs:
  - CSV: out/heatmap.csv  (customizable via --out-csv)
  - PNG: out/heatmap.png  (customizable via --out-png)

Usage:
  python scripts/heatmap.py \
    --db ingester/tropes.db \
    --work-id <WORK_UUID> \
    --top-n 20 \
    --out-csv out/heatmap.csv \
    --out-png out/heatmap.png
"""
from __future__ import annotations

import argparse
import csv
import math
import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


def fetch_scenes(conn: sqlite3.Connection, work_id: str) -> List[sqlite3.Row]:
    q = """
    SELECT s.id, s.idx AS scene_idx, c.idx AS chapter_idx
    FROM scene s
    LEFT JOIN chapter c ON c.id = s.chapter_id
    WHERE s.work_id=?
    ORDER BY s.idx ASC
    """
    conn.row_factory = sqlite3.Row
    return conn.execute(q, (work_id,)).fetchall()


def fetch_top_tropes(conn: sqlite3.Connection, work_id: str, top_n: int) -> List[sqlite3.Row]:
    # Frequency = number of distinct scenes a trope appears in (tie-break by total count)
    q = """
    SELECT t.id AS trope_id, t.name AS trope_name,
           COUNT(DISTINCT f.scene_id) AS scenes_covered,
           COUNT(*) AS total_hits
    FROM trope_finding f
    JOIN trope t ON t.id=f.trope_id
    WHERE f.work_id=?
    GROUP BY t.id
    ORDER BY scenes_covered DESC, total_hits DESC, t.name COLLATE NOCASE ASC
    LIMIT ?
    """
    conn.row_factory = sqlite3.Row
    return conn.execute(q, (work_id, top_n)).fetchall()


def fetch_conf_by_scene_trope(conn: sqlite3.Connection, work_id: str, trope_ids: List[str]) -> Dict[Tuple[str, str], float]:
    if not trope_ids:
        return {}
    placeholders = ",".join(["?"] * len(trope_ids))
    q = f"""
    SELECT f.scene_id, f.trope_id, MAX(f.confidence) AS max_conf
    FROM trope_finding f
    WHERE f.work_id=?
      AND f.trope_id IN ({placeholders})
    GROUP BY f.scene_id, f.trope_id
    """
    rows = conn.execute(q, (work_id, *trope_ids)).fetchall()
    out: Dict[Tuple[str, str], float] = {}
    for scene_id, trope_id, max_conf in rows:
        try:
            out[(scene_id, trope_id)] = float(max_conf or 0.0)
        except Exception:
            out[(scene_id, trope_id)] = 0.0
    return out


def label_for_scene(scene_idx: int, chapter_idx: int | None) -> str:
    if chapter_idx is None:
        return f"s{scene_idx}"
    return f"c{chapter_idx}:s{scene_idx}"


def write_csv(path: Path, scenes: List[sqlite3.Row], tropes: List[sqlite3.Row], mat: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ["scene_idx", "scene_label"] + [r["trope_name"] for r in tropes]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for i, srow in enumerate(scenes):
            scene_idx = int(srow["scene_idx"])
            label = label_for_scene(scene_idx, srow["chapter_idx"])
            w.writerow([scene_idx, label] + [f"{float(x):.3f}" for x in mat[i, :]])


def save_png(path: Path, title: str, scenes: List[sqlite3.Row], tropes: List[sqlite3.Row], mat: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    n_s, n_t = mat.shape if mat.size else (len(scenes), len(tropes))
    if n_s == 0 or n_t == 0:
        # Render a simple "no data" panel so the pipeline still produces a file.
        fig = plt.figure(figsize=(6, 2))
        plt.axis("off")
        plt.text(0.5, 0.5, "No data for heatmap", ha="center", va="center", fontsize=12)
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        return

    # Auto-size: ~0.42 in per scene row, ~0.45 in per trope col (with bounds)
    h = min(16, max(4, 0.42 * n_s))
    w = min(20, max(6, 0.45 * n_t))

    fig, ax = plt.subplots(figsize=(w, h))
    im = ax.imshow(mat, aspect="auto", vmin=0.0, vmax=1.0)

    # X labels: trope names
    xlabels = [r["trope_name"] for r in tropes]
    ax.set_xticks(np.arange(n_t), labels=xlabels, rotation=45, ha="right", fontsize=8)

    # Y labels: scene labels
    ylabels = [label_for_scene(int(r["scene_idx"]), r["chapter_idx"]) for r in scenes]
    ax.set_yticks(np.arange(n_s), labels=ylabels, fontsize=8)

    ax.set_xlabel("Trope")
    ax.set_ylabel("Scene")
    ax.set_title(f"Trope heatmap — {title}")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Max confidence per scene")

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Build a scene×trope heatmap (CSV + PNG).")
    ap.add_argument("--db", required=True)
    ap.add_argument("--work-id", required=True)
    ap.add_argument("--top-n", type=int, default=20, help="Top N tropes by frequency (default: 20)")
    ap.add_argument("--out-csv", default="out/heatmap.csv")
    ap.add_argument("--out-png", default="out/heatmap.png")
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row

    # Work title for the plot title
    wrow = conn.execute("SELECT title FROM work WHERE id=?", (args.work_id,)).fetchone()
    work_title = (wrow["title"] if wrow and wrow["title"] else args.work_id)

    scenes = fetch_scenes(conn, args.work_id)
    tropes = fetch_top_tropes(conn, args.work_id, args.top_n)

    trope_ids = [r["trope_id"] for r in tropes]
    conf_map = fetch_conf_by_scene_trope(conn, args.work_id, trope_ids)

    # Build matrix [n_scenes × n_tropes], default 0.0
    mat = np.zeros((len(scenes), len(tropes)), dtype=float)
    scene_index = {r["id"]: i for i, r in enumerate(scenes)}
    trope_index = {r["trope_id"]: j for j, r in enumerate(tropes)}

    for (scene_id, trope_id), v in conf_map.items():
        i = scene_index.get(scene_id)
        j = trope_index.get(trope_id)
        if i is not None and j is not None:
            try:
                mat[i, j] = max(0.0, min(1.0, float(v)))
            except Exception:
                mat[i, j] = 0.0

    # CSV + PNG
    write_csv(Path(args.out_csv), scenes, tropes, mat)
    save_png(Path(args.out_png), work_title, scenes, tropes, mat)

    print(f"[heatmap] wrote {args.out_csv} and {args.out_png}")


if __name__ == "__main__":
    main()
