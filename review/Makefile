# =========================
# Trope Miner — Review UI (gmake)
# =========================
.RECIPEPREFIX := >

# ---- Config (override on the command line) -------------------------------
DB                ?= ../ingester/tropes.db
HOST              ?= 127.0.0.1
PORT              ?= 5050

# Quick-win scripts expect these (adjust if you changed names)
OLLAMA_BASE_URL   ?= http://localhost:11434
EMBED_MODEL       ?= nomic-embed-text
CHROMA_HOST       ?= localhost
CHROMA_PORT       ?= 8000
CHUNK_COLLECTION  ?= trope-miner-v1-cos
TROPE_COLLECTION  ?= trope-catalog-nomic-cos

# Reporting / calibration knobs
WORK_ID      ?=
TITLE        ?=
APPLY        ?= 0          # set to 1 to apply edits in verify
K            ?= 10         # scenes to sample for calibration
THRESHOLD    ?= 0.50       # confidence threshold for calibration
IOU          ?= 0.30       # span IoU threshold for match
REPORTS_DIR  ?= ./reports

# ORDER: uncertain|newest|highest|random
ORDER ?= uncertain

MIN_CONF?=
MAX_CONF?=
LIMIT   ?= 0
FULL    ?= 0
COLOR   ?= 1
REVIEWER?=


# --- Sanitize config (strip stray whitespace) ------------------------------
REPORTS_DIR      := $(strip $(REPORTS_DIR))
DB               := $(strip $(DB))
WORK_ID          := $(strip $(WORK_ID))
TROPE_ID         := $(strip $(TROPE_ID))
TITLE            := $(strip $(TITLE))
ORDER            := $(strip $(ORDER))
MIN_CONF         := $(strip $(MIN_CONF))
MAX_CONF         := $(strip $(MAX_CONF))
LIMIT            := $(strip $(LIMIT))
FULL             := $(strip $(FULL))
COLOR            := $(strip $(COLOR))
REVIEWER         := $(strip $(REVIEWER))
THRESHOLD        := $(strip $(THRESHOLD))
IOU              := $(strip $(IOU))

# ---- Helpers --------------------------------------------------------------
SQLITE3 := sqlite3 $(DB)
NOW     := $(shell date +%Y%m%d-%H%M%S)

# ---- Phony ----------------------------------------------------------------
.PHONY: help whereis run open health warm dbcheck \
        reports reports-all verify calibrate clean \
        queue queue-open learn-thresholds learn-thresholds-json \
        queue-cli queue-cli-help

help:
> echo "Targets:"
> echo "  run             Start the Flask review app (debug) using TROPES_DB=$(DB)"
> echo "  open            Open http://$(HOST):$(PORT)"
> echo "  health          GET /healthz (app must be running)"
> echo "  warm            Hit '/' once to ensure review view exists (v_latest_human)"
> echo "  dbcheck         Quick counts: work/scene/finding"
> echo "  reports         HTML highlight report for one work (WORK_ID=<uuid> | TITLE=<exact>)"
> echo "  reports-all     Reports for all works → $(REPORTS_DIR)/"
> echo "  verify          Span verifier (embed-similarity + sentence snap)  [APPLY=1 to write]"
> echo "  calibrate       Mini calibration set (K scenes) → Precision/Recall/F1"
> echo "  clean           Remove reports/ and __pycache__"

whereis:
> echo "DB=$(DB)"
> echo "HOST=$(HOST) PORT=$(PORT)"
> echo "OLLAMA_BASE_URL=$(OLLAMA_BASE_URL)"
> echo "EMBED_MODEL=$(EMBED_MODEL)"
> echo "CHROMA_HOST=$(CHROMA_HOST) CHROMA_PORT=$(CHROMA_PORT)"
> echo "CHUNK_COLLECTION=$(CHUNK_COLLECTION)"
> echo "TROPE_COLLECTION=$(TROPE_COLLECTION)"
> echo "Scripts present?"
> echo -n "  scripts/report_highlights.py: " ; [ -f scripts/report_highlights.py ] && echo OK || echo MISSING
> echo -n "  scripts/verify_spans.py:      " ; [ -f scripts/verify_spans.py      ] && echo OK || echo MISSING
> echo -n "  scripts/calibrate_mini.py:    " ; [ -f scripts/calibrate_mini.py    ] && echo OK || echo MISSING
> echo "App files?"
> echo -n "  app.py:                       " ; [ -f app.py                       ] && echo OK || echo MISSING
> echo -n "  templates/scene.html:         " ; [ -f templates/scene.html         ] && echo OK || echo MISSING
> echo -n "  static/review.js:             " ; [ -f static/review.js             ] && echo OK || echo MISSING

# ---- Run the review app ---------------------------------------------------
run:
> echo "==> Using DB=$(DB)"
> TROPES_DB="$(DB)" FLASK_DEBUG=1 python3 app.py

open:
> python3 -c "import webbrowser; webbrowser.open('http://$(HOST):$(PORT)')"

health:
> curl -fsS "http://$(HOST):$(PORT)/healthz" || echo "healthz not reachable"

# Useful if you deep-link to /work or /scene before ever hitting '/'
# (index() creates v_latest_human in your current app.py).  :contentReference[oaicite:2]{index=2}
warm:
> curl -fsS "http://$(HOST):$(PORT)/" >/dev/null || echo "app not reachable"

# ---- DB sanity ------------------------------------------------------------
dbcheck:
> echo "works:"  ; $(SQLITE3) "SELECT COUNT(*) FROM work;"
> echo "scenes:" ; $(SQLITE3) "SELECT COUNT(*) FROM scene;"
> echo "findings:"; $(SQLITE3) "SELECT COUNT(*) FROM trope_finding;"

# ---- Reports (HTML with <mark> highlights) --------------------------------
reports:
> : $${WORK_ID:=$(TITLE)} ; \
> if [ -z "$$WORK_ID" ]; then \
>   echo "Usage: gmake reports WORK_ID=<uuid> | TITLE=<exact work title>"; exit 2; \
> fi; \
> mkdir -p "$(REPORTS_DIR)"; \
> python3 scripts/report_highlights.py --db "$(DB)" \
>   $(if $(WORK_ID),--work-id "$(WORK_ID)") $(if $(TITLE),--title "$(TITLE)") ; \
> echo "==> Reports in $(REPORTS_DIR)/"

reports-all:
> mkdir -p "$(REPORTS_DIR)"
> python3 scripts/report_highlights.py --db "$(DB)"


queue:
> $(MAKE) run DB=$(DB)

queue-open:
> python3 -c "import webbrowser, urllib.parse; q = urllib.parse.urlencode({}); webbrowser.open('http://$(HOST):$(PORT)/queue?' + q)"


learn-thresholds:
> python3 scripts/learn_thresholds.py --db "$(DB)" --min-count 6 --write-table

learn-thresholds-json:
> mkdir -p out
> python3 scripts/learn_thresholds.py --db "$(DB)" --min-count 6 --out out/thresholds.json --write-table
> echo "JSON written to review/out/thresholds.json"


# --- CLI queue -------------------------------------------------------------
queue-cli:
> : $${DB:?set DB=../ingester/tropes.db (or your path)}; :
> python3 scripts/queue_cli.py --db "$(DB)" \
>   $(if $(WORK_ID),--work-id "$(strip $(WORK_ID))") \
>   $(if $(TROPE_ID),--trope-id "$(strip $(TROPE_ID))") \
>   $(if $(ORDER),--order "$(strip $(ORDER))") \
>   $(if $(MIN_CONF),--min-conf $(strip $(MIN_CONF))) \
>   $(if $(MAX_CONF),--max-conf $(strip $(MAX_CONF))) \
>   --limit $(strip $(or $(LIMIT),0)) \
>   $(if $(filter 1,$(FULL)),--full,) \
>   $(if $(filter 0,$(COLOR)),--no-color,) \
>   $(if $(REVIEWER),--reviewer "$(strip $(REVIEWER))")

queue-cli-help:
> echo "Usage examples:"
> echo "  gmake queue-cli DB=../ingester/tropes.db"
> echo "  gmake queue-cli DB=../ingester/tropes.db WORK_ID=<uuid> ORDER=uncertain"
> echo "  gmake queue-cli DB=../ingester/tropes.db TROPE_ID=<uuid> MIN_CONF=0.30 MAX_CONF=0.70"
> echo "  gmake queue-cli DB=../ingester/tropes.db LIMIT=25 REVIEWER=larry"


# ---- Span verifier (Ollama + Chroma) -------------------------------------
verify:
> if [ -z "$(WORK_ID)" ]; then echo "Usage: gmake verify WORK_ID=<uuid> [APPLY=1]"; exit 2; fi
> OLLAMA_BASE_URL="$(OLLAMA_BASE_URL)" \
> EMBED_MODEL="$(EMBED_MODEL)" \
> CHROMA_HOST="$(CHROMA_HOST)" CHROMA_PORT="$(CHROMA_PORT)" \
> CHUNK_COLLECTION="$(CHUNK_COLLECTION)" TROPE_COLLECTION="$(TROPE_COLLECTION)" \
> python3 scripts/verify_spans.py --db "$(DB)" --work-id "$(WORK_ID)" $(if $(filter 1,$(APPLY)),--apply,)

# ---- Calibration (mini-set P/R/F1) ---------------------------------------
calibrate:
> python3 scripts/calibrate_mini.py --db "$(DB)" --k $(K) --threshold $(THRESHOLD) --iou $(IOU)

# ---- Cleanup --------------------------------------------------------------
clean:
> find . -name "__pycache__" -type d -prune -exec rm -rf {} +
> find . -name "*.pyc" -delete
> rm -rf "$(REPORTS_DIR)"
> echo "==> Cleaned $(REPORTS_DIR)/ and __pycache__"
