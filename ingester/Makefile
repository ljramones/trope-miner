# =========================
# Trope Miner Makefile (gmake, no tabs required)
# =========================
.RECIPEPREFIX := >

# ---- Config ----
DB           ?= ./tropes.db
WORK_ID      ?= 733b41a8-cd8a-4792-a2c1-4904ba994006
MODEL        ?= qwen2.5:7b-instruct
OLLAMA       ?= http://localhost:11434
THRESHOLD    ?= 0.25
TOP_K        ?= 8
TROPE_TOP_K  ?= 16

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
        report report-excerpts judge clean validate indexes vacuum sanity \
        export-findings report-findings report-findings-md grid \
        chroma-sanity chroma-reset-cos reembed reembed-tropes

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
> echo "  report/report-excerpts"
> echo "  judge, clean, validate, indexes, vacuum"
> echo "  export-findings    Export findings (OUT=..., FORMAT=csv|md, LIMIT=...)"
> echo "  report-findings    Judge then export CSV (timestamped)"
> echo "  report-findings-md Judge then export MD  (timestamped)"
> echo "  chroma-sanity      Verify Chroma collections use cosine"
> echo "  chroma-reset-cos   Recreate Chroma collections with cosine"
> echo "  reembed / reembed-tropes  Re-embed chunks / tropes"

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
\tpython scripts/seed_candidates_boundary.py --db "$(DB)" --work-id "$(WORK_ID)" --min-len 5 --max-per-trope 500


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

# Fresh judge: wipe previous findings for this WORK_ID first
judge-fresh:
> $(MAKE) clean
> $(MAKE) judge

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
>   $(SQLITE3) "CREATE INDEX IF NOT EXISTS idx_finding_work    ON trope_finding(work_id);" || true'
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

# Recreate Chroma collections as cosine (then re-embed)
chroma-reset-cos:
> python -c "code='''import chromadb
> c=chromadb.HttpClient(host=\"localhost\",port=8000)
> for name in [\"trope-miner-nomic-cos\",\"trope-catalog-nomic-cos\"]:
>     try:
>         c.delete_collection(name); print(\"deleted:\",name)
>     except Exception:
>         pass
>     c.create_collection(name, metadata={\"hnsw:space\":\"cosine\"})
>     print(\"created (cosine):\", name)
> '''; exec(code)"

# Re-embed narrative chunks into 'trope-miner-nomic-cos'
reembed:
> OLLAMA_BASE_URL=$(OLLAMA) python embedder.py \
>   --db "$(DB)" \
>   --collection trope-miner-nomic-cos \
>   --model nomic-embed-text \
>   --chroma-host localhost --chroma-port 8000 \
>   --batch-size 64 --limit 0

# Re-embed trope catalog into 'trope-catalog-nomic-cos'
reembed-tropes:
> OLLAMA_BASE_URL=$(OLLAMA) python embed_tropes.py
