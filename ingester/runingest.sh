#!/usr/bin/env bash
set -euo pipefail
trap 'echo "[error] line $LINENO: exit $?" >&2' ERR

# =========================================
# Trope Miner — One-shot ingest (A+B+C + 2A)
# =========================================
# A  = span verification & tightening
# B  = negation / anti-pattern downweight
# C  = confidence calibration (uses human labels if present)
# 2A = semantic seeding (recall without chaos)

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

# Trope catalog CSV (default lives in repo)
CSV="${TROPES_CSV:-$ROOT/tropes_data/trope_seed.csv}"

# Collections (names must match embed + judge + rerank)
CHUNK_COLL="${CHUNK_COLL:-trope-miner-v1-cos}"
TROPE_COLL="${TROPE_COLL:-trope-catalog-nomic-cos}"

# Per-work collection toggle (7A)
export PER_WORK_COLLECTIONS="${PER_WORK_COLLECTIONS:-0}"

# Make base names visible to the reranker/judge as env (reranker derives per-work name if enabled)
export CHUNK_COLLECTION="$CHUNK_COLL"
export TROPE_COLLECTION="$TROPE_COLL"

# Models
EMB_MODEL="${EMB_MODEL:-nomic-embed-text}"
REASONER_MODEL="${REASONER_MODEL:-llama3.1:8b}"

# Retrieval / priors (exported so reranker sees them)
export RERANK_TOP_K="${RERANK_TOP_K:-8}"
export RERANK_KEEP_M="${RERANK_KEEP_M:-3}"
export DOWNWEIGHT_NO_MENTION="${DOWNWEIGHT_NO_MENTION:-0.55}"
export SEM_SIM_THRESHOLD="${SEM_SIM_THRESHOLD:-0.36}"
TROPE_TOP_K="${TROPE_TOP_K:-16}"

# Threshold used by judge
THRESHOLD="${THRESHOLD:-0.25}"

# --- A: span verifier knobs ---
SPAN_VERIFIER_THRESHOLD="${SPAN_VERIFIER_THRESHOLD:-0.25}"
SPAN_VERIFIER_MAX_SENT="${SPAN_VERIFIER_MAX_SENT:-2}"

# --- B: anti-phrase seeding + negation/meta pass ---
ANTI_WINDOW="${ANTI_WINDOW:-60}"               # chars left/right for anti checks in seeding
NEGATION_MODE="${NEGATION_MODE:-downweight}"  # flag-only | downweight | delete
NEG_DOWNWEIGHT="${NEG_DOWNWEIGHT:-0.6}"
META_DOWNWEIGHT="${META_DOWNWEIGHT:-0.75}"
AA_DOWNWEIGHT="${AA_DOWNWEIGHT:-0.5}"

# --- (optional) alias expansion before seeding ---
EXPAND_ALIASES="${EXPAND_ALIASES:-0}"         # 1 to enable
ALIAS_MODEL="${ALIAS_MODEL:-$REASONER_MODEL}"

# --- 2A semantic seeding knobs ---
SEM_TAU="${SEM_TAU:-0.70}"                    # similarity threshold (1 - distance)
SEM_TOP_N="${SEM_TOP_N:-8}"                   # Chroma top-N per trope
SEM_PER_SCENE_CAP="${SEM_PER_SCENE_CAP:-3}"   # cap semantic seeds per (trope, scene)

# --- Optional analytics ---
GEN_HEATMAP="${GEN_HEATMAP:-0}"               # 1 to emit heatmap CSV/PNG
GEN_COOCCUR="${GEN_COOCCUR:-0}"               # 1 to emit co-occur CSV/GraphML/PNG
COOCC_TOP_N="${COOCC_TOP_N:-20}"
COOCC_MIN_WEIGHT="${COOCC_MIN_WEIGHT:-1}"

# --- Optional active learning (per-trope thresholds) ---
LEARN_THRESHOLDS="${LEARN_THRESHOLDS:-0}"     # 1 to run learn_thresholds.py if present

# ---------------- Sanity checks ----------------
[[ -f "$TEXT" ]] || { echo "ERROR: Missing TEXT file: $TEXT" >&2; exit 1; }
[[ -f "$CSV"  ]] || { echo "ERROR: Missing tropes CSV: $CSV" >&2; exit 1; }
command -v sqlite3 >/dev/null || { echo "ERROR: sqlite3 not found" >&2; exit 1; }
command -v python  >/dev/null || { echo "ERROR: python not found"  >&2; exit 1; }

# ---------------- 0) Fresh DB ------------------
echo "==> Resetting DB → $DB"
rm -f "$DB" "$DB-wal" "$DB-shm"
sqlite3 "$DB" < "$ROOT/sql/ingestion.sql" >/dev/null

# ---------------- 1) Load & embed trope catalog --------------
echo "==> Loading trope catalog CSV → SQLite… ($CSV)"
python "$ROOT/scripts/load_tropes.py" --db "$DB" --csv "$CSV"

echo "==> Embedding trope catalog → Chroma ($TROPE_COLL)…"
python "$ROOT/embed_tropes.py" \
  --db "$DB" \
  --collection "$TROPE_COLL" \
  --model "$EMB_MODEL" \
  --chroma-host "$CHROMA_HOST" \
  --chroma-port "$CHROMA_PORT" \
  --ollama-url "$OLLAMA_BASE_URL"

# ---------------- 1.5) (optional) expand aliases -------------
if [[ "$EXPAND_ALIASES" != "0" && -f "$ROOT/expand_trope_aliases.py" ]]; then
  echo "==> Expanding trope aliases with $ALIAS_MODEL …"
  python "$ROOT/expand_trope_aliases.py" --db "$DB" --model "$ALIAS_MODEL" || true
else
  echo "==> Skipping alias expansion (EXPAND_ALIASES=$EXPAND_ALIASES)"
fi

# ---------------- 2) Ingest the text -------------------------
python "$ROOT/ingestor_segmenter.py" ingest \
  --db "$DB" \
  --file "$TEXT" \
  --title "$TITLE" \
  --author "$AUTHOR"

# Newest work_id
WORK_ID="$(sqlite3 "$DB" 'SELECT id FROM work ORDER BY created_at DESC LIMIT 1;')"
echo "==> WORK_ID=$WORK_ID"

# Effective chunk collection (7A toggle)
if [[ "$PER_WORK_COLLECTIONS" == "1" ]]; then
  EFF_CHUNK_COLL="${CHUNK_COLL}__${WORK_ID}"
  echo "==> PER_WORK_COLLECTIONS=1 → using per-work collection: ${EFF_CHUNK_COLL}"
else
  EFF_CHUNK_COLL="$CHUNK_COLL"
  echo "==> PER_WORK_COLLECTIONS=0 → using global collection: ${EFF_CHUNK_COLL}"
fi

# ---------------- 3) Embed chunks ----------------------------
python "$ROOT/embedder.py" \
  --db "$DB" \
  --collection "$EFF_CHUNK_COLL" \
  --model "$EMB_MODEL" \
  --chroma-host "$CHROMA_HOST" \
  --chroma-port "$CHROMA_PORT" \
  --ollama-url "$OLLAMA_BASE_URL"

# ---------------- 4) Chroma sanity (non-fatal) ---------------
set +e
python "$ROOT/scripts/chroma_sanity.py" \
  --host "$CHROMA_HOST" \
  --port "$CHROMA_PORT" \
  --collections "$EFF_CHUNK_COLL" "$TROPE_COLL" \
  --probe \
  --self-threshold 1e-3
SANITY_RC=$?
set -e
if [ "$SANITY_RC" -ne 0 ]; then
  echo "[warn] chroma sanity returned $SANITY_RC — continuing pipeline"
fi

# ---------------- 5) Seed boundary gazetteer (with anti) ----
echo "==> Seeding boundary matches (ANTI_WINDOW=${ANTI_WINDOW})…"
if python "$ROOT/scripts/seed_candidates_boundary.py" -h 2>&1 | grep -q -- '--anti-window'; then
  python "$ROOT/scripts/seed_candidates_boundary.py" \
    --db "$DB" \
    --work-id "$WORK_ID" \
    --anti-window "$ANTI_WINDOW"
else
  # Back-compat: script without --anti-window
  python "$ROOT/scripts/seed_candidates_boundary.py" \
    --db "$DB" \
    --work-id "$WORK_ID"
fi

# ---------------- 5.5) Seed semantic candidates (2A) --------
if [[ -f "$ROOT/scripts/seed_candidates_semantic.py" ]]; then
  echo "==> Seeding semantic matches (tau=${SEM_TAU}, topN=${SEM_TOP_N}, cap/scene=${SEM_PER_SCENE_CAP})…"
  python "$ROOT/scripts/seed_candidates_semantic.py" \
    --db "$DB" \
    --work-id "$WORK_ID" \
    --collection "$EFF_CHUNK_COLL" \
    --chroma-host "$CHROMA_HOST" \
    --chroma-port "$CHROMA_PORT" \
    --embed-model "$EMB_MODEL" \
    --ollama-url "$OLLAMA_BASE_URL" \
    --tau "$SEM_TAU" \
    --top-n "$SEM_TOP_N" \
    --per-scene-cap "$SEM_PER_SCENE_CAP"
else
  echo "==> Skipping semantic seeding (scripts/seed_candidates_semantic.py not found)"
fi

# ---------------- 6) Judge scenes (retrieval + rerank + sanity) ---
python "$ROOT/trope_miner_tools.py" judge-scenes \
  --db "$DB" \
  --work-id "$WORK_ID" \
  --collection "$CHUNK_COLL" \
  --chroma-host "$CHROMA_HOST" \
  --chroma-port "$CHROMA_PORT" \
  --embed-model "$EMB_MODEL" \
  --reasoner-model "$REASONER_MODEL" \
  --ollama-url "$OLLAMA_BASE_URL" \
  --trope-collection "$TROPE_COLL" \
  --trope-top-k "$TROPE_TOP_K" \
  --top-k "$RERANK_TOP_K" \
  --threshold "$THRESHOLD"

# ---------------- 6.5) A: Verify / tighten spans -------------
echo "==> Span verification (threshold=${SPAN_VERIFIER_THRESHOLD}, max-sent=${SPAN_VERIFIER_MAX_SENT})…"
python "$ROOT/scripts/span_verifier.py" \
  --db "$DB" \
  --work-id "$WORK_ID" \
  --model "$EMB_MODEL" \
  --ollama-url "$OLLAMA_BASE_URL" \
  --threshold "$SPAN_VERIFIER_THRESHOLD" \
  --max-sentences "$SPAN_VERIFIER_MAX_SENT"

# ---------------- 6.7) B: Negation/meta/anti-* pass ----------
if [[ -f "$ROOT/scripts/verifier_pass.py" ]]; then
  echo "==> Negation/meta verifier (mode=${NEGATION_MODE})…"
  python "$ROOT/scripts/verifier_pass.py" \
    --db "$DB" \
    --work-id "$WORK_ID" \
    --mode "$NEGATION_MODE" \
    --window 40 \
    --neg-downweight "$NEG_DOWNWEIGHT" \
    --meta-downweight "$META_DOWNWEIGHT" \
    --antialias-downweight "$AA_DOWNWEIGHT"
else
  echo "==> Skipping negation/meta verifier (scripts/verifier_pass.py not found)"
fi

# ---------------- 7) Reports & exports -----------------------
echo "==> Exporting findings + HTML report…"
python "$ROOT/scripts/export_findings.py" \
  --db "$DB" \
  --work-id "$WORK_ID" \
  --format csv \
  --out "$OUT_DIR/findings.csv"

python "$ROOT/scripts/report_html.py" \
  --db "$DB" \
  --work-id "$WORK_ID" \
  --out "$OUT_DIR/report_${WORK_ID}.html"

# Optional: support/sanity snapshot markdown
python "$ROOT/scripts/support_report.py" \
  --db "$DB" \
  --work-id "$WORK_ID" \
  --format md \
  --threshold "${THRESHOLD}" \
  --out "$OUT_DIR/support_${WORK_ID}.md" >/dev/null 2>&1 || true

# ---------------- 7.5) Optional analytics --------------------
if [[ "$GEN_HEATMAP" != "0" && -f "$ROOT/scripts/heatmap.py" ]]; then
  echo "==> Heatmap (scenes × top-20 tropes)…"
  python "$ROOT/scripts/heatmap.py" \
    --db "$DB" \
    --work-id "$WORK_ID" \
    --csv "$OUT_DIR/heatmap_${WORK_ID}.csv" \
    --png "$OUT_DIR/heatmap_${WORK_ID}.png" \
    --top 20 || true
fi

if [[ "$GEN_COOCCUR" != "0" && -f "$ROOT/scripts/cooccur.py" ]]; then
  echo "==> Co-occurrence graph…"
  python "$ROOT/scripts/cooccur.py" \
    --db "$DB" \
    --work-id "$WORK_ID" \
    --threshold "$THRESHOLD" \
    --out-csv "$OUT_DIR/cooccur_${WORK_ID}.csv" \
    --out-graphml "$OUT_DIR/cooccur_${WORK_ID}.graphml" \
    --png "$OUT_DIR/cooccur_${WORK_ID}.png" \
    --top-n "$COOCC_TOP_N" \
    --min-weight "$COOCC_MIN_WEIGHT" || true
fi

# ---------------- 8) C: Calibration (if human labels exist) ---
echo "==> Checking for human decisions (accept/reject)…"
LABELED_COUNT="$(sqlite3 "$DB" "SELECT COUNT(*) FROM v_latest_human WHERE decision IN ('accept','reject');")"

if [ "${LABELED_COUNT}" -gt 0 ]; then
  echo "==> Found ${LABELED_COUNT} labeled findings; running calibration…"
  CALIB_CSV="$OUT_DIR/calibration.csv"
  CALIB_PNG="$OUT_DIR/calibration.png"
  set +e
  python "$ROOT/scripts/calibrate_threshold.py" \
    --db "$DB" \
    --out "$CALIB_CSV" \
    --plot "$CALIB_PNG" \
    --work-id "$WORK_ID" \
    --bins "$CALIB_BINS" \
    --step "$CALIB_STEP" \
    --objective "$CALIB_OBJECTIVE" \
    --min-precision "$CALIB_MIN_PREC" \
    --min-recall "$CALIB_MIN_REC"
  CALIB_RC=$?
  set -e

  if [ "$CALIB_RC" -eq 0 ] && [ -f "$CALIB_CSV" ]; then
    REC_THR="$(grep -m1 '^recommended_threshold,' "$CALIB_CSV" | awk -F',' '{print $2}' | tr -d '[:space:]')"
    if [[ -n "${REC_THR:-}" && "$REC_THR" != "None" ]]; then
      echo "==> Calibration recommends THRESHOLD ≈ ${REC_THR}"
      echo "    Use it on next runs, e.g.: THRESHOLD=${REC_THR} ./runingest.sh"
    else
      echo "[calib] No recommended threshold found in $CALIB_CSV (insufficient data?)."
    fi
  else
    echo "[calib] Skipping (no output or error)."
  fi
else
  echo "==> No human labels yet; skipping calibration."
fi

# ---------------- 9) Optional: learn per-trope thresholds -----
if [[ "$LEARN_THRESHOLDS" != "0" && -f "$ROOT/scripts/learn_thresholds.py" ]]; then
  echo "==> Active learning: fitting per-trope thresholds…"
  python "$ROOT/scripts/learn_thresholds.py" \
    --db "$DB" \
    --out "$OUT_DIR/trope_thresholds.csv" || true
fi

echo "Done. Findings → $OUT_DIR/findings.csv"
echo "HTML report → $OUT_DIR/report_${WORK_ID}.html"
