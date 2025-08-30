#!/usr/bin/env bash
set -euo pipefail

# --- Config ---
ROOT="$(cd "$(dirname "$0")" && pwd)"
DB="$ROOT/tropes.db"
OUT_DIR="$ROOT/out"
mkdir -p "$OUT_DIR"

TEXT="${TEXT_PATH:-$ROOT/../data/TheGirl.txt}"
TITLE="${TITLE:-The Girl}"
AUTHOR="${AUTHOR:-Larry Mitchell}"

CHROMA_HOST="${CHROMA_HOST:-localhost}"
CHROMA_PORT="${CHROMA_PORT:-8000}"
OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://localhost:11434}"

# Let user override with TROPES_CSV; default to ingester/tropes_data/trope_seed.csv
CSV="${TROPES_CSV:-$ROOT/tropes_data/trope_seed.csv}"

CHUNK_COLL="${CHUNK_COLL:-trope-miner-v1-cos}"
TROPE_COLL="${TROPE_COLL:-trope-catalog-nomic-cos}"

EMB_MODEL="${EMB_MODEL:-nomic-embed-text}"
REASONER_MODEL="${REASONER_MODEL:-llama3.1:8b}"

# --- Sanity checks ---
[[ -f "$TEXT" ]] || { echo "ERROR: Missing TEXT file: $TEXT"; exit 1; }
[[ -f "$CSV"  ]] || { echo "ERROR: Missing tropes CSV: $CSV"; exit 1; }

# --- 0) Ensure schema (quiet) ---
sqlite3 "$DB" < "$ROOT/sql/ingestion.sql" >/dev/null

# --- 1) Load & embed trope catalog ---
echo "Loading trope catalog CSV → SQLite… ($CSV)"
python "$ROOT/scripts/load_tropes.py" --db "$DB" --csv "$CSV"

echo "Embedding trope catalog → Chroma ($TROPE_COLL)…"
python "$ROOT/embed_tropes.py" \
  --db "$DB" \
  --collection "$TROPE_COLL" \
  --model "$EMB_MODEL" \
  --chroma-host "$CHROMA_HOST" \
  --chroma-port "$CHROMA_PORT" \
  --ollama-url "$OLLAMA_BASE_URL"

# --- 2) Ingest the text ---
python "$ROOT/ingestor_segmenter.py" ingest \
  --db "$DB" \
  --file "$TEXT" \
  --title "$TITLE" \
  --author "$AUTHOR"

# Newest work_id
WORK_ID="$(sqlite3 "$DB" 'SELECT id FROM work ORDER BY created_at DESC LIMIT 1;')"
echo "WORK_ID=$WORK_ID"

# --- 3) Embed text chunks into Chroma ---
python "$ROOT/embedder.py" \
  --db "$DB" \
  --collection "$CHUNK_COLL" \
  --model "$EMB_MODEL" \
  --chroma-host "$CHROMA_HOST" \
  --chroma-port "$CHROMA_PORT" \
  --ollama-url "$OLLAMA_BASE_URL"

# --- 4) Chroma sanity (count/space/dim; self-probe) ---
# Sanity is advisory. Don't fail the whole run if it complains.
set +e
python "$ROOT/scripts/chroma_sanity.py" \
  --host "$CHROMA_HOST" \
  --port "$CHROMA_PORT" \
  --collections "$CHUNK_COLL" "$TROPE_COLL" \
  --probe \
  --self-threshold 1e-3
SANITY_RC=$?
set -e
if [ "$SANITY_RC" -ne 0 ]; then
  echo "[warn] chroma sanity returned $SANITY_RC — continuing pipeline"
fi


# --- 5) Seed boundary-aware gazetteer candidates ---
python "$ROOT/scripts/seed_candidates_boundary.py" \
  --db "$DB" \
  --work-id "$WORK_ID"

# --- 6) Judge scenes (retrieval + LLM) ---
python "$ROOT/trope_miner_tools.py" judge-scenes \
  --db "$DB" \
  --work-id "$WORK_ID" \
  --collection "$CHUNK_COLL" \
  --chroma-host "$CHROMA_HOST" \
  --chroma-port "$CHROMA_PORT" \
  --embed-model "$EMB_MODEL" \
  --reasoner-model "$REASONER_MODEL" \
  --ollama-url "$OLLAMA_BASE_URL" \
  --trope-collection "$TROPE_COLL"

# 6.5) Verify / tighten spans
python "$ROOT/scripts/span_verifier.py" \
  --db "$DB" \
  --work-id "$WORK_ID" \
  --model "$EMB_MODEL" \
  --ollama-url "$OLLAMA_BASE_URL" \
  --threshold 0.25 \
  --max-sentences 2


# --- 7) Export findings ---
python "$ROOT/scripts/export_findings.py" \
  --db "$DB" \
  --work-id "$WORK_ID" \
  --format csv \
  --out "$OUT_DIR/findings.csv"

python "$ROOT/scripts/report_html.py" \
  --db "$DB" \
  --work-id "$WORK_ID" \
  --out "$OUT_DIR/report_${WORK_ID}.html"


echo "Done. Findings → $OUT_DIR/findings.csv"
