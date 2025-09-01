#!/usr/bin/env python3
"""
Batch-ingest a folder of texts and produce one HTML report per work.

What it does per file:
  1) Ingest/segment the text → SQLite (work/chapters/scenes/chunks).
  2) Embed new chunks → Chroma.
  3) Seed candidates (boundary + semantic, if present).
  4) Judge scenes (retrieval → rerank → sanity → LLM).
  5) Write HTML report to out/report_<WORK_ID>.html

It also ensures the trope catalog exists and is embedded once up-front.

Notes
  • This script DOES NOT reset the DB. It appends new works.
  • For simplicity, batching assumes a **global chunk collection**
    (PER_WORK_COLLECTIONS=0). If you’re using per-work collections,
    run works one-by-one or adapt embedder to filter by work_id.

Usage:
  python scripts/batch_ingest.py \
    --db ./tropes.db \
    --input-dir ../data \
    --glob "*.txt" \
    --out ./out \
    --csv ./tropes_data/trope_seed.csv \
    --threshold 0.25

Env it respects (fallbacks shown):
  CHROMA_HOST=localhost  CHROMA_PORT=8000
  OLLAMA_BASE_URL=http://localhost:11434
  EMB_MODEL=nomic-embed-text
  REASONER_MODEL=llama3.1:8b
  CHUNK_COLLECTION=trope-miner-v1-cos
  TROPE_COLLECTION=trope-catalog-nomic-cos
  RERANK_TOP_K=8  TROPE_TOP_K=16
  SEM_TAU=0.70  SEM_TOP_N=8  SEM_PER_SCENE_CAP=3
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

# ---------- paths & helpers ----------

ROOT = Path(__file__).resolve().parent.parent  # .../ingester
SCRIPTS = ROOT / "scripts"

def sh(cmd: List[str]) -> None:
    subprocess.run(cmd, check=True)

def ensure_db_and_tropes(db: Path, csv: Path, chroma_host: str, chroma_port: int,
                         ollama: str, emb_model: str, trope_coll: str) -> None:
    """Create schema if needed; ensure trope catalog loaded & embedded."""
    need_schema = not db.exists()
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    try:
        if need_schema:
            print(f"==> Creating schema → {db}")
            sql_path = ROOT / "sql" / "ingestion.sql"
            sql = sql_path.read_text(encoding="utf-8")
            conn.executescript(sql)
            conn.commit()

        # is trope table populated?
        try:
            n = conn.execute("SELECT COUNT(*) FROM trope").fetchone()[0]
        except sqlite3.OperationalError:
            n = 0

        if n == 0:
            print(f"==> Loading trope catalog CSV → SQLite… ({csv})")
            sh([sys.executable, str(SCRIPTS / "load_tropes.py"),
                "--db", str(db), "--csv", str(csv)])

        # Always (re)embed catalog to Chroma (safe, idempotent)
        print(f"==> Embedding trope catalog → Chroma ({trope_coll})…")
        sh([sys.executable, str(ROOT / "embed_tropes.py"),
            "--db", str(db),
            "--collection", trope_coll,
            "--model", emb_model,
            "--chroma-host", chroma_host,
            "--chroma-port", str(chroma_port),
            "--ollama-url", ollama])
    finally:
        conn.close()

def newest_work_id(conn: sqlite3.Connection) -> str:
    r = conn.execute("SELECT id FROM work ORDER BY created_at DESC LIMIT 1").fetchone()
    return r[0] if r else ""

def ingest_one(db: Path, text_path: Path, title: str, author: str) -> str:
    """Runs the segmenter and returns the new work_id."""
    sh([sys.executable, str(ROOT / "ingestor_segmenter.py"), "ingest",
        "--db", str(db),
        "--file", str(text_path),
        "--title", title,
        "--author", author])
    with sqlite3.connect(str(db)) as conn:
        return newest_work_id(conn)

def embed_chunks(db: Path, chunk_coll: str, chroma_host: str, chroma_port: int,
                 ollama: str, emb_model: str) -> None:
    print(f"==> Embedding chunks → {chunk_coll}")
    sh([sys.executable, str(ROOT / "embedder.py"),
        "--db", str(db),
        "--collection", chunk_coll,
        "--model", emb_model,
        "--chroma-host", chroma_host,
        "--chroma-port", str(chroma_port),
        "--ollama-url", ollama])

def seed_boundary(db: Path, work_id: str, anti_window: int) -> None:
    print(f"==> Seeding boundary matches (ANTI_WINDOW={anti_window})…")
    script = SCRIPTS / "seed_candidates_boundary.py"
    args = ["--db", str(db), "--work-id", work_id]
    # add anti-window if supported
    try:
        help_txt = subprocess.run([sys.executable, str(script), "-h"],
                                  check=True, capture_output=True, text=True).stderr + \
                   subprocess.run([sys.executable, str(script), "-h"],
                                  check=False, capture_output=True, text=True).stdout
    except Exception:
        help_txt = ""
    if "--anti-window" in help_txt:
        args += ["--anti-window", str(anti_window)]
    sh([sys.executable, str(script), *args])

def seed_semantic(db: Path, work_id: str, chunk_coll: str, chroma_host: str, chroma_port: int,
                  emb_model: str, ollama: str, tau: float, top_n: int, per_scene_cap: int) -> None:
    script = SCRIPTS / "seed_candidates_semantic.py"
    if not script.exists():
        print("==> Skipping semantic seeding (scripts/seed_candidates_semantic.py not found)")
        return
    print(f"==> Seeding semantic matches (tau={tau}, topN={top_n}, cap/scene={per_scene_cap})…")
    sh([sys.executable, str(script),
        "--db", str(db),
        "--work-id", work_id,
        "--collection", chunk_coll,
        "--chroma-host", chroma_host,
        "--chroma-port", str(chroma_port),
        "--embed-model", emb_model,
        "--ollama-url", ollama,
        "--tau", str(tau),
        "--top-n", str(top_n),
        "--per-scene-cap", str(per_scene_cap)])

def judge(db: Path, work_id: str, chunk_coll: str, chroma_host: str, chroma_port: int,
          emb_model: str, reasoner_model: str, ollama: str,
          trope_coll: str, trope_top_k: int, top_k: int, threshold: float) -> None:
    print("==> Judging scenes…")
    sh([sys.executable, str(ROOT / "trope_miner_tools.py"), "judge-scenes",
        "--db", str(db),
        "--work-id", work_id,
        "--collection", chunk_coll,
        "--chroma-host", chroma_host,
        "--chroma-port", str(chroma_port),
        "--embed-model", emb_model,
        "--reasoner-model", reasoner_model,
        "--ollama-url", ollama,
        "--trope-collection", trope_coll,
        "--trope-top-k", str(trope_top_k),
        "--top-k", str(top_k),
        "--threshold", str(threshold)])

def report_html(db: Path, work_id: str, out_dir: Path) -> Path:
    out = out_dir / f"report_{work_id}.html"
    sh([sys.executable, str(SCRIPTS / "report_html.py"),
        "--db", str(db), "--work-id", work_id, "--out", str(out)])
    return out

# ---------- CLI ----------

def parse_args():
    p = argparse.ArgumentParser(description="Batch ingest a folder of texts and produce HTML reports.")
    p.add_argument("--db", required=True, type=Path)
    p.add_argument("--input-dir", required=True, type=Path)
    p.add_argument("--glob", default="*.txt", help="Glob pattern to match (default: *.txt)")
    p.add_argument("--out", type=Path, default=Path("./out"))
    p.add_argument("--csv", type=Path, default=None, help="Trope CSV (defaults to ingester/tropes_data/trope_seed.csv)")
    p.add_argument("--author", default="Unknown Author")
    p.add_argument("--title-mode", choices=["stem", "filename"], default="stem",
                   help="Use file stem or whole filename as title (default: stem)")
    # services / models / collections
    p.add_argument("--chroma-host", default=os.getenv("CHROMA_HOST", "localhost"))
    p.add_argument("--chroma-port", type=int, default=int(os.getenv("CHROMA_PORT", "8000")))
    p.add_argument("--ollama-url", default=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    p.add_argument("--embed-model", default=os.getenv("EMB_MODEL", "nomic-embed-text"))
    p.add_argument("--reasoner-model", default=os.getenv("REASONER_MODEL", "llama3.1:8b"))
    p.add_argument("--chunk-coll", default=os.getenv("CHUNK_COLLECTION", os.getenv("CHUNK_COLL", "trope-miner-v1-cos")))
    p.add_argument("--trope-coll", default=os.getenv("TROPE_COLLECTION", os.getenv("TROPE_COLL", "trope-catalog-nomic-cos")))
    # judge knobs
    p.add_argument("--top-k", type=int, default=int(os.getenv("RERANK_TOP_K", "8")))
    p.add_argument("--trope-top-k", type=int, default=int(os.getenv("TROPE_TOP_K", "16")))
    p.add_argument("--threshold", type=float, default=float(os.getenv("THRESHOLD", "0.25")))
    # seeding knobs
    p.add_argument("--anti-window", type=int, default=int(os.getenv("ANTI_WINDOW", "60")))
    p.add_argument("--sem-tau", type=float, default=float(os.getenv("SEM_TAU", "0.70")))
    p.add_argument("--sem-top-n", type=int, default=int(os.getenv("SEM_TOP_N", "8")))
    p.add_argument("--sem-per-scene-cap", type=int, default=int(os.getenv("SEM_PER_SCENE_CAP", "3")))
    return p.parse_args()

def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    csv_path = args.csv or (ROOT / "tropes_data" / "trope_seed.csv")
    ensure_db_and_tropes(args.db, csv_path, args.chroma_host, args.chroma_port,
                         args.ollama_url, args.embed_model, args.trope_coll)

    # Safety: batching expects global collection
    if os.getenv("PER_WORK_COLLECTIONS", "0") == "1":
        print("[warn] PER_WORK_COLLECTIONS=1 is not supported in batch mode. "
              "Proceeding with the global collection:", args.chunk_coll)

    files = sorted(args.input_dir.rglob(args.glob))
    if not files:
        print(f"No files matching {args.glob} under {args.input_dir}")
        return

    summary: List[Tuple[str, str, Path]] = []  # (filename, work_id, report_path)

    for path in files:
        if not path.is_file():
            continue
        title = path.stem if args.title_mode == "stem" else path.name
        print(f"\n=== Processing: {path.name} (title='{title}') ===")
        try:
            work_id = ingest_one(args.db, path, title, args.author)
            print(f"==> WORK_ID={work_id}")

            embed_chunks(args.db, args.chunk_coll, args.chroma_host, args.chroma_port,
                         args.ollama_url, args.embed_model)

            seed_boundary(args.db, work_id, args.anti_window)
            seed_semantic(args.db, work_id, args.chunk_coll, args.chroma_host, args.chroma_port,
                          args.embed_model, args.ollama_url, args.sem_tau, args.sem_top_n, args.sem_per_scene_cap)

            judge(args.db, work_id, args.chunk_coll, args.chroma_host, args.chroma_port,
                  args.embed_model, args.reasoner_model, args.ollama_url,
                  args.trope_coll, args.trope_top_k, args.top_k, args.threshold)

            rpt = report_html(args.db, work_id, args.out)
            summary.append((path.name, work_id, rpt))
        except subprocess.CalledProcessError as e:
            print(f"[error] step failed for {path.name}: {e}")
        except Exception as e:
            print(f"[error] unexpected error for {path.name}: {e}")

    # Summary
    if summary:
        print("\n=== Batch complete ===")
        for fname, wid, rpt in summary:
            print(f"- {fname} → work_id={wid} → {rpt}")
    else:
        print("\nNo reports produced.")

if __name__ == "__main__":
    main()
