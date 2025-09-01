# =========================
# Trope Miner Makefile (gmake, no tabs required)
# =========================
.RECIPEPREFIX := >

# ---- Config ----

# Collections + Chroma endpoint (can be overridden on the CLI)
CHUNK_COLL  ?= trope-miner-v1-cos
TROPE_COLL  ?= trope-catalog-nomic-cos
CHROMA_HOST ?= localhost
CHROMA_PORT ?= 8000

DB           ?= ./tropes.db
WORK_ID      ?= 733b41a8-cd8a-4792-a2c1-4904ba994006
MODEL        ?= qwen2.5:7b-instruct
OLLAMA       ?= http://localhost:11434
THRESHOLD    ?= 0.25
TOP_K        ?= 8
TROPE_TOP_K  ?= 16

# Two-stage rerank & sanity
export RERANK_TOP_K       ?= 8
export RERANK_KEEP_M      ?= 3
export DOWNWEIGHT_NO_MENTION ?= 0.55
export SEM_SIM_THRESHOLD  ?= 0.36

# Review API (Flask)
API_HOST ?= 127.0.0.1
API_PORT ?= 5050
API_BASE := http://$(API_HOST):$(API_PORT)

# Verifier (span_verifier.py) defaults
VERIFIER_MODEL         ?= nomic-embed-text
VERIFIER_THRESHOLD     ?= 0.32
VERIFIER_ALPHA         ?= 0.7
VERIFIER_MIN_GAIN      ?= 0.05
VERIFIER_MAX_SENTENCES ?= 2
VERIFIER_MAX_CHARS     ?= 280

# Schema / seed file
SCHEMA       ?= ./sql/ingestion.sql
CSV          ?= ./trope_seed.csv

# strip accidental trailing spaces
SCHEMA := $(strip $(SCHEMA))
CSV    := $(strip $(CSV))

# ---- Helpers ----
SQLITE3 := sqlite3 $(DB)
NOW     := $(shell date +%Y%m%d-%H%M%S)

# ---- Phony ----
.PHONY: help whereis quickstart all all-boundary init-db init-db-fresh backup-db \
        load-tropes reload-tropes aliases dry-aliases seed seed-boundary \
        report report-excerpts report-html judge clean validate indexes vacuum sanity \
        export-findings report-findings report-findings-md grid \
        chroma-sanity chroma-reset-cos reembed reembed-tropes \
        support-report \
        calibrate calibrate-plot span-snap-dry span-snap-apply \
        span-verify span-verify-dry \
        judge-verify-report \
        test-smoke \
        api-accept api-accept-latest api-edit-span api-edit-span-latest \
        seed-boundary-no-anti seed-semantic \
        heatmap learn-thresholds \
        cooccur

help:
> echo "Targets:"
> echo "  quickstart        init-db-fresh -> reload-tropes -> all-boundary -> judge"
> echo "  whereis           Show config and script presence"
> echo "  init-db           Apply schema if missing (idempotent)"
> echo "  init-db-fresh     Remove DB then apply schema"
> echo "  backup-db         Copy DB to tropes.db.$(NOW).bak"
> echo "  load-tropes       Upsert tropes from CSV ($(CSV))"
> echo "  reload-tropes     Wipe 'trope' then load CSV"
> echo "  all               aliases -> seed -> report"
> echo "  all-boundary      aliases -> seed-boundary -> report"
> echo "  aliases/dry-aliases"
> echo "  seed/seed-boundary"
> echo "  report/report-excerpts/report-html"
> echo "  judge, judge-verify-report, clean, validate, indexes, vacuum"
> echo "  export-findings    Export findings (OUT=..., FORMAT=csv|md, LIMIT=...)"
> echo "  report-findings    Judge then export CSV (timestamped)"
> echo "  report-findings-md Judge then export MD  (timestamped)"
> echo "  chroma-sanity      Verify Chroma collections use cosine"
> echo "  chroma-reset-cos   Recreate Chroma collections with cosine"
> echo "  reembed / reembed-tropes  Re-embed chunks / tropes"
> echo "  support-report     Per-scene support/sanity report (md)"
> echo "  calibrate / calibrate-plot"
> echo "  seed-boundary-no-anti  Gazetteer seeding without anti-phrase blocking"
> echo "  span-snap-dry / span-snap-apply / span-verify(-dry)"
> echo "  test-smoke         Minimal pytest checks against DB"
> echo "  api-*              Review UI API smoketests"

whereis:
> echo "DB=$(DB)"
> echo "WORK_ID=$(WORK_ID)"
> echo "MODEL=$(MODEL)"
> echo "OLLAMA=$(OLLAMA)"
> echo "SCHEMA=$(SCHEMA)"
> echo "CSV=$(CSV)"
> echo "expand_trope_aliases.py: " $$([ -f expand_trope_aliases.py ] && echo OK || echo MISSING)
> echo "trope_miner_tools.py:    " $$([ -f trope_miner_tools.py    ] && echo OK || echo MISSING)
> echo "scripts/load_tropes.py:  " $$([ -f scripts/load_tropes.py  ] && echo OK || echo MISSING)
> echo "scripts/seed_candidates_boundary.py: " $$([ -f scripts/seed_candidates_boundary.py ] && echo OK || echo MISSING)

# Pipelines
quickstart:
> $(MAKE) init-db-fresh
> $(MAKE) reload-tropes
> $(MAKE) all-boundary
> $(MAKE) judge

all: aliases seed report
all-boundary: aliases seed-boundary report

# ---- DB init / seed ----
init-db:
> need_main=$$($(SQLITE3) "SELECT 1 FROM sqlite_master WHERE type='table' AND name='work';"); \
> if [ -z "$$need_main" ]; then \
>   echo '==> Applying schema $(SCHEMA)…'; \
>   test -f "$(SCHEMA)" || { echo "Schema file not found: $(SCHEMA)"; exit 1; }; \
>   $(SQLITE3) ".read $(SCHEMA)"; \
> else \
>   echo '==> Schema already present; skipping.'; \
> fi; \
> $(MAKE) indexes

init-db-fresh:
> echo "==> Removing existing DB + WAL/SHM (if any)…"
> rm -f "$(DB)" "$(DB)-wal" "$(DB)-shm"
> echo "==> Applying schema $(SCHEMA)…"
> test -f "$(SCHEMA)" || { echo "Schema file not found: $(SCHEMA)"; exit 1; }
> $(SQLITE3) ".read $(SCHEMA)"
> echo "==> Schema applied."
> $(MAKE) indexes

backup-db:
> if [ -f "$(DB)" ]; then \
>   cp "$(DB)" "$(DB).$(NOW).bak"; \
>   echo "==> Backup: $(DB).$(NOW).bak"; \
> else \
>   echo "No DB found at $(DB)"; exit 1; \
> fi

load-tropes:
> echo "==> Loading tropes from $(CSV) into $(DB) (upsert)…"
> test -f "$(CSV)" || { echo "CSV not found: $(CSV)"; exit 1; }
> python scripts/load_tropes.py --db "$(DB)" --csv "$(CSV)"
> echo "==> Row count now:"
> $(SQLITE3) "SELECT COUNT(*) FROM trope;"

reload-tropes:
> echo "==> RELOADING tropes: clearing table, then loading from $(CSV)…"
> test -f "$(CSV)" || { echo "CSV not found: $(CSV)"; exit 1; }
> python scripts/load_tropes.py --db "$(DB)" --csv "$(CSV)" --clear
> echo "==> Row count now:"
> $(SQLITE3) "SELECT COUNT(*) FROM trope;"

# ---- Alias expansion ----
aliases:
> echo "==> Expanding trope aliases with $(MODEL)…"
> python expand_trope_aliases.py --db "$(DB)" --model "$(MODEL)"

dry-aliases:
> echo "==> DRY RUN: expanding 10 aliases (no DB writes)"
> python expand_trope_aliases.py --db "$(DB)" --model "$(MODEL)" --dry-run --limit 10

# ---- Seeding ----
seed:
> echo "==> Seeding gazetteer candidates for $(WORK_ID)…"
> python trope_miner_tools.py seed-candidates --db "$(DB)" --work-id "$(WORK_ID)" || true
> echo "==> Candidate count:"
> $(SQLITE3) "SELECT COUNT(*) FROM trope_candidate WHERE work_id='$(WORK_ID)';"

seed-boundary:
> python scripts/seed_candidates_boundary.py --db "$(DB)" --work-id "$(WORK_ID)" --min-len 5 --max-per-trope 500

support-report:
> : $${DB:?set DB=./tropes.db}; : $${WORK_ID:?set WORK_ID=<uuid>}; \
> script="scripts/support_report.py"; \
> [ -f "$$script" ] || script="ingester/scripts/support_report.py"; \
> python "$$script" --db "$$DB" --work-id "$$WORK_ID" --format md --threshold $${THRESHOLD:-0.55} --out out/support.$(NOW).md

seed-boundary-no-anti:
> python scripts/seed_candidates_boundary.py \
>   --db "$(DB)" --work-id "$(WORK_ID)" \
>

# Semantic seeding (Chroma top-N per trope with sim >= TAU)
seed-semantic:
> : $${DB:?set DB=./tropes.db}; : $${WORK_ID:?set WORK_ID=<uuid>}; \
> : $${CHUNK_COLL:=trope-miner-v1-cos}; : $${CHROMA_HOST:=localhost}; : $${CHROMA_PORT:=8000}; \
> : $${TAU:=0.70}; : $${TOP_N:=8}; : $${PER_SCENE_CAP:=3}; \
> python scripts/seed_candidates_semantic.py \
>   --db "$$DB" --work-id "$$WORK_ID" \
>   --collection "$$CHUNK_COLL" \
>   --chroma-host "$$CHROMA_HOST" --chroma-port "$$CHROMA_PORT" \
>   --embed-model nomic-embed-text --ollama-url "$${OLLAMA:-http://localhost:11434}" \
>   --tau "$$TAU" --top-n "$$TOP_N" --per-scene-cap "$$PER_SCENE_CAP"

# ---- Reporting ----
report:
> echo "==> Tropes with non-empty aliases:"
> $(SQLITE3) "SELECT COUNT(*) FROM trope WHERE aliases IS NOT NULL AND TRIM(aliases)<>'';"
> echo "==> Top tropes by candidate hits:"
> $(SQLITE3) "SELECT t.name, COUNT(*) AS hits \
>              FROM trope_candidate c \
>              JOIN trope t ON t.id=c.trope_id \
>              WHERE c.work_id='$(WORK_ID)' \
>              GROUP BY t.id \
>              ORDER BY hits DESC \
>              LIMIT 15;"

report-excerpts:
> echo '==> Recent candidate excerpts:'
> $(SQLITE3) "SELECT \
>                t.name AS trope, \
>                c.alias AS alias_hit, \
>                c.surface AS surface_form, \
>                substr(k.text, (c.start - k.char_start) + 1, (c.end - c.start)) AS excerpt \
>              FROM trope_candidate c \
>              JOIN trope t ON t.id = c.trope_id \
>              JOIN chunk k ON k.id = c.chunk_id \
>              WHERE c.work_id='$(WORK_ID)' \
>                AND c.start >= k.char_start \
>                AND c.end   <= k.char_end \
>              ORDER BY c.created_at DESC \
>              LIMIT 10;"

report-html:
> mkdir -p out
> : $${DB:?set DB=./tropes.db}; : $${WORK_ID:?set WORK_ID=<uuid>}; \
> script="scripts/report_html.py"; \
> [ -f "$$script" ] || script="ingester/scripts/report_html.py"; \
> OUT="out/report_$${WORK_ID}.html"; \
> echo "==> Writing $$OUT"; \
> python "$$script" --db "$$DB" --work-id "$$WORK_ID" --out "$$OUT"

# ---- Judge ----
judge:
> echo "==> Running judge-scenes with $(MODEL)…"
> python trope_miner_tools.py judge-scenes \
>   --db "$(DB)" --work-id "$(WORK_ID)" \
>   --collection trope-miner-nomic-cos --embed-model nomic-embed-text \
>   --reasoner-model "$(MODEL)" --ollama-url "$(OLLAMA)" \
>   --top-k "$(TOP_K)" --threshold "$(THRESHOLD)" \
>   --trope-collection trope-catalog-nomic-cos --trope-top-k "$(TROPE_TOP_K)"
> echo "==> Recent findings (chunk-derived excerpts):"
> $(SQLITE3) "SELECT \
>                t.name, printf('%.2f', f.confidence) AS conf, \
>                substr(k.text, (f.evidence_start - k.char_start) + 1, (f.evidence_end - f.evidence_start)) AS excerpt \
>              FROM trope_finding f \
>              JOIN trope t ON t.id=f.trope_id \
>              JOIN chunk k ON k.work_id=f.work_id \
>                            AND f.evidence_start>=k.char_start AND f.evidence_end<=k.char_end \
>              WHERE f.work_id='$(WORK_ID)' \
>              ORDER BY f.created_at DESC \
>              LIMIT 10;"

# One-shot: judge -> verify spans -> HTML + CSV report -> support snapshot
judge-verify-report:
> $(MAKE) judge
> $(MAKE) span-verify
> $(MAKE) verify-negation
> $(MAKE) report-html
> $(MAKE) report-findings
> $(MAKE) support-report

# Fresh judge: wipe previous findings for this WORK_ID first
judge-fresh:
> $(MAKE) clean
> $(MAKE) judge

learn-thresholds:
> : $${DB:?set DB=./tropes.db}; \
> python ingester/scripts/learn_thresholds.py --db "$$DB" --out out/trope_thresholds.csv

dedupe:
> $(SQLITE3) "DELETE FROM trope_candidate WHERE rowid NOT IN (SELECT MIN(rowid) FROM trope_candidate GROUP BY work_id, trope_id, start, end);"
> $(SQLITE3) "DELETE FROM trope_finding  WHERE rowid NOT IN (SELECT MIN(rowid) FROM trope_finding  GROUP BY work_id, trope_id, evidence_start, evidence_end);"
> echo "==> De-dupe complete."

# De-duplicate findings for this WORK_ID (keep first occurrence per logical key)
dedupe-findings:
> echo "==> De-duplicating trope_finding for $(WORK_ID)…"
> $(SQLITE3) "DELETE FROM trope_finding \
>              WHERE work_id='$(WORK_ID)' AND rowid NOT IN ( \
>                SELECT MIN(rowid) FROM trope_finding \
>                WHERE work_id='$(WORK_ID)' \
>                GROUP BY scene_id, trope_id, evidence_start, evidence_end, model \
>              );"
> echo "==> Count after dedupe:"
> $(SQLITE3) "SELECT COUNT(*) FROM trope_finding WHERE work_id='$(WORK_ID)';"

# (Optional) enforce uniqueness going forward
ensure-unique-findings:
> $(SQLITE3) "CREATE UNIQUE INDEX IF NOT EXISTS uq_finding_key \
>              ON trope_finding(work_id, scene_id, trope_id, evidence_start, evidence_end, model);"

reseed-boundary:
> echo "==> Reseeding gazetteer candidates for $(WORK_ID)…"
> $(SQLITE3) "DELETE FROM trope_candidate WHERE work_id='$(WORK_ID)' AND source='gazetteer';"
> $(MAKE) seed-boundary

# ---- Maintenance ----
clean:
> echo "==> Cleaning candidates + findings for $(WORK_ID)…"
> $(SQLITE3) "DELETE FROM trope_candidate WHERE work_id='$(WORK_ID)';"
> $(SQLITE3) "DELETE FROM trope_finding  WHERE work_id='$(WORK_ID)';"
> echo "==> Done."

validate:
> python - <<'PY'
> import sqlite3, json
> conn=sqlite3.connect("$(DB)")
> c=conn.cursor()
> bad=0
> for (id,aliases) in c.execute("SELECT id,aliases FROM trope WHERE aliases IS NOT NULL AND TRIM(aliases)<>''"):
>     try:
>         a=json.loads(aliases); assert isinstance(a,list)
>     except Exception:
>         bad+=1
> print("bad_alias_rows:", bad)
> conn.close()
> PY

indexes:
> echo "==> Ensuring indexes (conditional)…"
> $(SQLITE3) "SELECT name FROM sqlite_master WHERE type='table' AND name='trope_candidate';" | grep -q trope_candidate && \
>   $(SQLITE3) "CREATE INDEX IF NOT EXISTS idx_candidate_work  ON trope_candidate(work_id);" || true
> $(SQLITE3) "SELECT name FROM sqlite_master WHERE type='table' AND name='trope_candidate';" | grep -q trope_candidate && \
>   $(SQLITE3) "CREATE INDEX IF NOT EXISTS idx_candidate_trope ON trope_candidate(trope_id);" || true
> $(SQLITE3) "SELECT name FROM sqlite_master WHERE type='table' AND name='chunk';" | grep -q chunk && \
>   $(SQLITE3) "CREATE INDEX IF NOT EXISTS idx_chunk_work_span ON chunk(work_id, char_start, char_end);" || true
> $(SQLITE3) "SELECT name FROM sqlite_master WHERE type='table' AND name='trope_finding';" | grep -q trope_finding && \
>   $(SQLITE3) "CREATE INDEX IF NOT EXISTS idx_finding_work    ON trope_finding(work_id);" || true
> $(SQLITE3) "CREATE UNIQUE INDEX IF NOT EXISTS uq_candidate_span ON trope_candidate(work_id, trope_id, start, end);" || true
> $(SQLITE3) "CREATE UNIQUE INDEX IF NOT EXISTS uq_finding_span  ON trope_finding(work_id, trope_id, evidence_start, evidence_end);" || true
> echo "==> Indexes ensured."

vacuum:
> $(SQLITE3) "ANALYZE; VACUUM;"

sanity:
> echo "==> Sanity check on $(DB)"
> $(SQLITE3) "SELECT name FROM sqlite_master WHERE type='table' AND name IN ('work','trope','trope_candidate','trope_finding') ORDER BY name;" | tr '\n' ' ' | sed 's/$$/\n/' | awk '{print "Tables present:", $$0}'
> echo "- work columns:"
> $(SQLITE3) "PRAGMA table_info(work);" | awk -F'|' '{print "   ", $$2, $$3}'
> echo "- trope columns:"
> $(SQLITE3) "PRAGMA table_info(trope);" | awk -F'|' '{print "   ", $$2, $$3}'
> echo "- trope_candidate columns:"
> $(SQLITE3) "PRAGMA table_info(trope_candidate);" | awk -F'|' '{print "   ", $$2, $$3}'
> echo "- trope_finding columns:"
> $(SQLITE3) "PRAGMA table_info(trope_finding);" | awk -F'|' '{print "   ", $$2, $$3}'
> echo "- created_at on trope_finding?"
> if $(SQLITE3) "PRAGMA table_info(trope_finding);" | awk -F'|' '{print $$2}' | grep -qx 'created_at'; then \
>   echo "   ✅ trope_finding.created_at present"; \
> else \
>   echo "   ❌ trope_finding.created_at MISSING"; exit 1; \
> fi

# Export current findings to OUT (creates parent dir)
# Usage:
#   gmake export-findings OUT=out/findings.csv FORMAT=csv LIMIT=200
#   gmake export-findings OUT=out/findings.md  FORMAT=md
export-findings:
> : $${OUT:?"OUT=<path> required (e.g., out/findings.csv)"}; :
> FORMAT=$${FORMAT:-csv}; LIMIT=$${LIMIT:-0}; \
>   mkdir -p "$$(dirname "$$OUT")"; \
>   echo "==> Exporting findings for $(WORK_ID) to $$OUT (format=$$FORMAT, limit=$$LIMIT)"; \
>   python scripts/export_findings.py \
>     --db "$(DB)" --work-id "$(WORK_ID)" \
>     --format $$FORMAT --out "$$OUT" $$(test "$$LIMIT" -gt 0 && printf -- "--limit %s" "$$LIMIT")

# Sweep a few thresholds; prints counts after each run
# Usage:
#   gmake grid THS="0.20 0.25 0.30"
grid:
> THS=$${THS:-"0.20 0.25 0.30"}; for TH in $$THS; do \
>   echo "==> THRESHOLD $$TH"; \
>   python trope_miner_tools.py judge-scenes \
>     --db "$(DB)" --work-id "$(WORK_ID)" \
>     --collection trope-miner-nomic-cos --embed-model nomic-embed-text \
>     --reasoner-model "$(MODEL)" --ollama-url "$(OLLAMA)" \
>     --top-k "$(TOP_K)" --threshold $$TH \
>     --trope-collection trope-catalog-nomic-cos --trope-top-k "$(TROPE_TOP_K)"; \
>   $(SQLITE3) "SELECT COUNT(*) AS findings FROM trope_finding WHERE work_id='$(WORK_ID)';"; \
> done

# ---- Reporting ----
report-findings:
> mkdir -p out
> echo "==> Exporting findings to out/findings.$(NOW).csv"
> python scripts/export_findings.py \
>   --db "$(DB)" --work-id "$(WORK_ID)" \
>   --format csv --out "out/findings.$(NOW).csv"

report-findings-md:
> mkdir -p out
> echo "==> Exporting findings to out/findings.$(NOW).md"
> python scripts/export_findings.py \
>   --db "$(DB)" --work-id "$(WORK_ID)" \
>   --format md --out "out/findings.$(NOW).md"

# Verify collections are cosine and reachable
# (requires scripts/chroma_sanity.py)
chroma-sanity:
> python scripts/chroma_sanity.py \
>   --host localhost --port 8000 \
>   --collections trope-miner-nomic-cos trope-catalog-nomic-cos

# Recreate Chroma collections as cosine (quote-safe)
chroma-reset-cos:
> python3 -c "import chromadb; c=chromadb.HttpClient(host='$(CHROMA_HOST)',port=$(CHROMA_PORT)); \
>   [c.delete_collection(n) for n in ['$(CHUNK_COLL)','$(TROPE_COLL)'] if n in [x.name for x in c.list_collections()]]; \
>   c.create_collection('$(CHUNK_COLL)', metadata={'hnsw:space':'cosine'}); \
>   c.create_collection('$(TROPE_COLL)',  metadata={'hnsw:space':'cosine'}); \
>   print('reset:', [x.name for x in c.list_collections()])"

# Re-embed narrative chunks into CHUNK_COLL
# Use DB relative to *this* dir (ingester/)
reembed:
> OLLAMA_BASE_URL=$(OLLAMA) python3 embedder.py \
>   --db "$(DB)" \
>   --collection "$(CHUNK_COLL)" \
>   --model nomic-embed-text \
>   --chroma-host "$(CHROMA_HOST)" --chroma-port "$(CHROMA_PORT)" \
>   --batch-size 64 --limit 0

# Re-embed trope catalog into TROPE_COLL
reembed-tropes:
> OLLAMA_BASE_URL=$(OLLAMA) python3 embed_tropes.py \
>   --db "$(DB)" \
>   --collection "$(TROPE_COLL)" \
>   --model nomic-embed-text \
>   --chroma-host "$(CHROMA_HOST)" --chroma-port "$(CHROMA_PORT)"

# ---- New: Calibration & Span tools ----
calibrate:
> mkdir -p out
> python scripts/calibrate_threshold.py --db "$(DB)" --out "out/calibration.csv"

calibrate-plot:
> mkdir -p out
> python scripts/calibrate_threshold.py --db "$(DB)" --out "out/calibration.csv" --plot "out/calibration.png"

span-snap-dry:
> python scripts/span_snap.py --db "$(DB)" --dry-run

span-snap-apply:
> python scripts/span_snap.py --db "$(DB)" --apply

span-verify-dry:
> python scripts/span_verifier.py \
>   --db "$(DB)" --work-id "$(WORK_ID)" \
>   --ollama-url "$(OLLAMA)" --model "$(VERIFIER_MODEL)" \
>   --threshold "$(VERIFIER_THRESHOLD)" --alpha "$(VERIFIER_ALPHA)" \
>   --min-gain "$(VERIFIER_MIN_GAIN)" --max-sentences "$(VERIFIER_MAX_SENTENCES)" \
>   --max-chars "$(VERIFIER_MAX_CHARS)" --dry-run

span-verify:
> python scripts/span_verifier.py \
>   --db "$(DB)" --work-id "$(WORK_ID)" \
>   --ollama-url "$(OLLAMA)" --model "$(VERIFIER_MODEL)" \
>   --threshold "$(VERIFIER_THRESHOLD)" --alpha "$(VERIFIER_ALPHA)" \
>   --min-gain "$(VERIFIER_MIN_GAIN)" --max-sentences "$(VERIFIER_MAX_SENTENCES)" \
>   --max-chars "$(VERIFIER_MAX_CHARS)"

# ---- New: Pytest smoke ----
test-smoke:
> TEST_FILE="tests/test_pipeline_smoke.py"; \
> [ -f "$$TEST_FILE" ] || TEST_FILE="../tests/test_pipeline_smoke.py"; \
> if [ -f "$$TEST_FILE" ]; then \
>   TROPES_DB="$(DB)" pytest -q "$$TEST_FILE"; \
> else \
>   echo "Note: create tests/test_pipeline_smoke.py to enable this."; \
> fi

# ---- New: API smoke helpers (Flask must be running) ----
api-accept:
> : $${FINDING_ID:?"FINDING_ID required (usage: gmake api-accept FINDING_ID=<uuid> DECISION=accept|reject)"}; \
> DECISION=$${DECISION:-accept}; \
> echo "==> POST /api/decision finding_id=$$FINDING_ID decision=$$DECISION"; \
> curl -sS -X POST "$(API_BASE)/api/decision" \
>   -H 'Content-Type: application/json' \
>   -d "$$(printf '{"finding_id":"%s","decision":"%s"}' "$$FINDING_ID" "$$DECISION")"; echo

api-accept-latest:
> latest=$$($(SQLITE3) "SELECT id FROM trope_finding WHERE work_id='$(WORK_ID)' ORDER BY created_at DESC LIMIT 1;"); \
> if [ -z "$$latest" ]; then \
>   latest=$$($(SQLITE3) "SELECT id FROM trope_finding WHERE work_id='$(WORK_ID)' ORDER BY rowid DESC LIMIT 1;"); \
> fi; \
> if [ -z "$$latest" ]; then echo "No findings for WORK_ID=$(WORK_ID)"; exit 1; fi; \
> $(MAKE) api-accept FINDING_ID="$$latest" DECISION=$${DECISION:-accept}

api-edit-span:
> : $${FINDING_ID:?"FINDING_ID required (usage: gmake api-edit-span FINDING_ID=<uuid> START=<n> END=<n>)"}; \
> : $${START:?"START required"}; : $${END:?"END required"}; \
> echo "==> POST /api/edit_span finding_id=$$FINDING_ID start=$$START end=$$END"; \
> curl -sS -X POST "$(API_BASE)/api/edit_span" \
>   -H 'Content-Type: application/json' \
>   -d "$$(printf '{"finding_id":"%s","start":%s,"end":%s}' "$$FINDING_ID" "$$START" "$$END")"; echo

api-edit-span-latest:
> latest=$$($(SQLITE3) "SELECT id FROM trope_finding WHERE work_id='$(WORK_ID)' ORDER BY created_at DESC LIMIT 1;"); \
> if [ -z "$$latest" ]; then \
>   latest=$$($(SQLITE3) "SELECT id FROM trope_finding WHERE work_id='$(WORK_ID)' ORDER BY rowid DESC LIMIT 1;"); \
> fi; \
> if [ -z "$$latest" ]; then echo "No findings for WORK_ID=$(WORK_ID)"; exit 1; fi; \
> : $${START:?"START required"}; : $${END:?"END required"}; \
> $(MAKE) api-edit-span FINDING_ID="$$latest" START="$$START" END="$$END"

verify-negation-dry:
> python scripts/verifier_pass.py --db "$(DB)" --work-id "$(WORK_ID)" --mode flag-only --window 40 --dry-run

verify-negation:
> python scripts/verifier_pass.py --db "$(DB)" --work-id "$(WORK_ID)" --mode downweight --window 40 --neg-downweight 0.6 --meta-downweight 0.75 --antialias-downweight 0.5

heatmap:
> mkdir -p out
> : $${DB:?set DB=./tropes.db}; : $${WORK_ID:?set WORK_ID=<uuid>}; \
> python scripts/heatmap.py \
>   --db "$$DB" --work-id "$$WORK_ID" \
>   --top-n $${TOP_N:-20} \
>   --out-csv out/heatmap.csv \
>   --out-png out/heatmap.png

cooccur:
> : $${DB:?set DB=./tropes.db}; : $${WORK_ID:?set WORK_ID=<uuid>}; \
> TH=$${THRESHOLD:-0.25}; NOW=$$(date +%Y%m%d-%H%M%S); mkdir -p out; \
> python scripts/cooccur.py --db "$$DB" --work-id "$$WORK_ID" \
>   --threshold "$$TH" \
>   --out-csv "out/cooccur.$$NOW.csv" \
>   --out-graphml "out/cooccur.$$NOW.graphml" \
>   --png "out/cooccur.$$NOW.png" --top-n 20 --min-weight 2