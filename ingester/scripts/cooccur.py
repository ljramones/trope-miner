#!/usr/bin/env python3
"""
Co-occurrence graph of tropes per work.

- For each scene, collect the set of tropes with adjusted confidence >= THRESHOLD
  (i.e., trope_finding.confidence).
- For every unordered pair within that scene-set, increment an edge count.
- Export:
    * CSV (src_id, src_name, dst_id, dst_name, weight)
    * GraphML (undirected, node label + count, edge weight)
    * Optional PNG chord-like diagram for top-N frequent tropes.

Usage:
  python scripts/cooccur.py \
    --db ingester/tropes.db --work-id <WORK_ID> \
    --threshold 0.55 \
    --out-csv out/cooccur.csv \
    --out-graphml out/cooccur.graphml \
    --png out/cooccur.png --top-n 20 --min-weight 2

Notes:
- If you want to restrict to *accepted* human labels only, pass --human-only.
- If matplotlib isn't installed, PNG generation is skipped with a warning.
"""
from __future__ import annotations

import argparse
import os
import sqlite3
from collections import defaultdict, Counter
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional
import html as _html

# ----------------------------- data fetch -----------------------------

def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    cur = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1", (name,)
    )
    return cur.fetchone() is not None

def fetch_scene_tropes(
    conn: sqlite3.Connection,
    work_id: str,
    threshold: float,
    human_only: bool = False,
) -> Tuple[Dict[str, Set[str]], Dict[str, str]]:
    """
    Returns:
      scene_to_tropes: {scene_id: {trope_id, ...}}
      trope_name: {trope_id: name}
    Only counts a trope once per scene (set semantics).
    """
    # Pull trope names first
    name_rows = conn.execute("SELECT id, name FROM trope").fetchall()
    trope_name = {r[0]: r[1] or r[0] for r in name_rows}

    # Findings ≥ threshold
    if human_only and _table_exists(conn, "v_latest_human"):
        rows = conn.execute(
            """
            SELECT DISTINCT f.scene_id, f.trope_id
            FROM trope_finding f
            JOIN v_latest_human h ON h.finding_id = f.id AND h.decision = 'accept'
            WHERE f.work_id = ? AND f.confidence >= ?
            """,
            (work_id, threshold),
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT DISTINCT f.scene_id, f.trope_id
            FROM trope_finding f
            WHERE f.work_id = ? AND f.confidence >= ?
            """,
            (work_id, threshold),
        ).fetchall()

    scene_to_tropes: Dict[str, Set[str]] = defaultdict(set)
    for scene_id, trope_id in rows:
        if trope_id:
            scene_to_tropes[str(scene_id)].add(str(trope_id))
    return scene_to_tropes, trope_name

# ----------------------------- exports -----------------------------

def write_csv(
    out_csv: Path,
    edges: Dict[Tuple[str, str], int],
    trope_name: Dict[str, str],
) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8") as f:
        f.write("src_id,src_name,dst_id,dst_name,weight\n")
        for (a, b), w in sorted(edges.items(), key=lambda kv: (-kv[1], kv[0])):
            f.write(
                f"{a},{csv_safe(trope_name.get(a, a))},"
                f"{b},{csv_safe(trope_name.get(b, b))},"
                f"{w}\n"
            )
    print(f"[cooccur] wrote CSV: {out_csv.resolve()}")

def csv_safe(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    if any(c in s for c in [",", '"', "\n"]):
        s = '"' + s.replace('"', '""') + '"'
    return s

def write_graphml(
    out_graphml: Path,
    edges: Dict[Tuple[str, str], int],
    node_counts: Dict[str, int],
    trope_name: Dict[str, str],
) -> None:
    out_graphml.parent.mkdir(parents=True, exist_ok=True)
    # Simple GraphML with node label+count, edge weight
    lines = []
    lines.append('<?xml version="1.0" encoding="UTF-8"?>')
    lines.append('<graphml xmlns="http://graphml.graphdrawing.org/xmlns" '
                 'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
                 'xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns '
                 'http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">')
    lines.append('<key id="d0" for="node" attr.name="label" attr.type="string"/>')
    lines.append('<key id="d1" for="node" attr.name="count" attr.type="int"/>')
    lines.append('<key id="d2" for="edge" attr.name="weight" attr.type="int"/>')
    lines.append('<graph id="G" edgedefault="undirected">')

    # Nodes
    for nid, cnt in sorted(node_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        label = trope_name.get(nid, nid)
        lines.append(f'  <node id="{xml_safe(nid)}">')
        lines.append(f'    <data key="d0">{xml_safe(label)}</data>')
        lines.append(f'    <data key="d1">{cnt}</data>')
        lines.append('  </node>')

    # Edges
    eid = 0
    for (a, b), w in sorted(edges.items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f'  <edge id="e{eid}" source="{xml_safe(a)}" target="{xml_safe(b)}">')
        lines.append(f'    <data key="d2">{w}</data>')
        lines.append('  </edge>')
        eid += 1

    lines.append('</graph>')
    lines.append('</graphml>')

    out_graphml.write_text("\n".join(lines), encoding="utf-8")
    print(f"[cooccur] wrote GraphML: {out_graphml.resolve()}")

def xml_safe(s: str) -> str:
    return _html.escape("" if s is None else str(s), quote=True)

# ----------------------------- PNG chord -----------------------------

def write_png_chord(
    out_png: Path,
    node_counts: Dict[str, int],
    edges: Dict[Tuple[str, str], int],
    trope_name: Dict[str, str],
    top_n: int,
    min_weight: int,
) -> None:
    """
    Very simple chord/arc plot:
      - take top-N nodes by count
      - draw arcs on a circle
      - draw straight chords (alpha scaled by weight)
    """
    try:
        import math
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as e:
        print(f"[cooccur] matplotlib not available; skipping PNG ({e})")
        return

    # Keep only top-N nodes
    top_nodes = [nid for nid, _ in sorted(node_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:top_n]]
    idx = {nid: i for i, nid in enumerate(top_nodes)}
    if len(top_nodes) < 2:
        print("[cooccur] not enough nodes for a chord diagram; skipping PNG.")
        return

    # Build adjacency for shown nodes
    pairs = {k: v for k, v in edges.items()
             if k[0] in idx and k[1] in idx and v >= max(1, int(min_weight))}
    if not pairs:
        print("[cooccur] no edges above min_weight for top-N; skipping PNG.")
        return

    # Layout on circle
    n = len(top_nodes)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    xy = np.c_[np.cos(angles), np.sin(angles)]

    fig = plt.figure(figsize=(8, 8), dpi=150)
    ax = fig.add_subplot(111)
    ax.set_aspect("equal")
    ax.axis("off")

    # Draw nodes (points + labels)
    for nid, i in idx.items():
        x, y = xy[i]
        ax.plot([x], [y], marker="o")  # default style; avoid specifying colors
        label = trope_name.get(nid, nid)
        ax.text(x * 1.1, y * 1.1, label, ha="center", va="center", fontsize=8)

    # Edge alpha scaling
    wvals = list(pairs.values())
    wmin, wmax = min(wvals), max(wvals)
    def alpha_of(w):
        if wmax == wmin:
            return 0.5
        return 0.2 + 0.8 * (w - wmin) / (wmax - wmin)

    # Draw chords (straight lines)
    for (a, b), w in pairs.items():
        i, j = idx[a], idx[b]
        x1, y1 = xy[i]
        x2, y2 = xy[j]
        ax.plot([x1, x2], [y1, y2], linewidth=1 + 1.5 * alpha_of(w), alpha=alpha_of(w))

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"[cooccur] wrote PNG: {out_png.resolve()}")

# ----------------------------- main -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Compute trope co-occurrence per work and export CSV/GraphML/(optional)PNG.")
    ap.add_argument("--db", required=True)
    ap.add_argument("--work-id", required=True)
    ap.add_argument("--threshold", type=float, default=float(os.getenv("THRESHOLD", "0.25")),
                    help="Adjusted-confidence cutoff (default from $THRESHOLD or 0.25).")
    ap.add_argument("--human-only", action="store_true",
                    help="Only count findings with latest human decision = accept (requires v_latest_human).")

    ap.add_argument("--out-csv", type=str, default=None, help="Path to write CSV (edges).")
    ap.add_argument("--out-graphml", type=str, default=None, help="Path to write GraphML.")
    ap.add_argument("--png", type=str, default=None, help="Path to write a simple chord diagram PNG.")
    ap.add_argument("--top-n", type=int, default=20, help="Top-N nodes (by scene count) to include in PNG.")
    ap.add_argument("--min-weight", type=int, default=1, help="Min co-occurrence weight to draw in PNG.")
    args = ap.parse_args()

    db = Path(args.db)
    if not db.exists():
        raise SystemExit(f"DB not found: {db}")

    conn = sqlite3.connect(str(db))

    # 1) scene sets
    scene2set, trope_name = fetch_scene_tropes(conn, args.work_id, args.threshold, args.human_only)

    # 2) edges (unordered pairs)
    edges: Dict[Tuple[str, str], int] = defaultdict(int)
    node_counts: Counter = Counter()
    for scene_id, tropes in scene2set.items():
        if not tropes or len(tropes) < 2:
            # still count nodes so degree/GraphML has all present tropes
            for t in tropes:
                node_counts[t] += 1
            continue
        for t in tropes:
            node_counts[t] += 1
        for a, b in combinations(sorted(tropes), 2):
            edges[(a, b)] += 1

    # 3) outputs
    if args.out_csv:
        write_csv(Path(args.out_csv), edges, trope_name)
    if args.out_graphml:
        write_graphml(Path(args.out_graphml), edges, dict(node_counts), trope_name)
    if args.png:
        write_png_chord(Path(args.png), dict(node_counts), edges, trope_name, args.top_n, args.min_weight)

    conn.close()

if __name__ == "__main__":
    main()
