Here’s a clean, ready-to-drop-in **RUNME.md** that matches your current pipeline (A+B+C + 2A semantic seeding), plus optional heatmap & co-occurrence steps.

---

# RUNME.md — Trope Miner (One-Shot Ingest + Review)

This doc shows how to run the full pipeline end-to-end and how to tweak it with environment variables. It covers:

* **A**: span verification & tightening
* **B**: negation / anti-pattern handling (seed-time + verify pass)
* **C**: confidence calibration (based on your review accepts/rejects)
* **2A**: **semantic seeding** (higher recall with a similarity cap)

---

## Quick start

```bash
# 1) From repo root
cd ingester
pip install -r requirements.txt

# 2) Make sure services are up:
#    - Chroma at CHROMA_HOST:CHROMA_PORT (default localhost:8000)
#    - Ollama at OLLAMA_BASE_URL (default http://localhost:11434)
#      and that the models exist locally:
#        ollama pull nomic-embed-text
#        ollama pull llama3.1:8b   # or qwen2.5:7b-instruct

# 3) Run the one-shot ingest (resets DB, runs A+B+2A, writes reports)
./runingest.sh
```

Open the Review UI (optional, to accept/reject/edit):

```bash
cd ../review
export TROPES_DB=../ingester/tropes.db
python app.py
# → http://127.0.0.1:5050
```

---

## What `runingest.sh` does

1. Creates a fresh SQLite DB (`ingester/tropes.db`) from `sql/ingestion.sql`.
2. Loads the trope catalog CSV and embeds to Chroma (**TROPE** collection).
3. Ingests & segments your text into chapters/scenes/chunks.
4. Embeds chunks to Chroma (**CHUNK** collection).
5. Seeds **boundary-aware** candidates (with **anti-phrase / anti-X** suppression).
6. **Semantic seeding (2A)**: queries chunk vectors per work and inserts high-sim seeds.
7. Judges scenes with **per-work retrieval → LLM rerank → sanity priors**.
8. **A**: span verification/tightening and **B**: negation/meta/anti-\* pass.
9. Exports `out/findings.csv`, `out/report_<WORK_ID>.html`, and (best effort) `out/support_<WORK_ID>.md`.
10. **C**: If you’ve accepted/rejected in the UI, runs calibration and writes `out/calibration.csv/.png`.

---

## Core environment variables (tune without editing scripts)

> Set inline, e.g.:
>
> ```bash
> THRESHOLD=0.55 REASONER_MODEL=qwen2.5:7b-instruct ./runingest.sh
> ```

### Data & services

| Variable          | Default                               | Purpose                 |
| ----------------- | ------------------------------------- | ----------------------- |
| `TEXT_PATH`       | `../data/TheGirl.txt`                 | Path to the input work. |
| `TITLE`           | `The Girl`                            | Title stored in DB.     |
| `AUTHOR`          | `Larry Mitchell`                      | Author stored in DB.    |
| `TROPES_CSV`      | `ingester/tropes_data/trope_seed.csv` | Trope catalog CSV.      |
| `CHROMA_HOST`     | `localhost`                           | Chroma host.            |
| `CHROMA_PORT`     | `8000`                                | Chroma port.            |
| `OLLAMA_BASE_URL` | `http://localhost:11434`              | Ollama endpoint.        |

### Models & vector collections

| Variable         | Default                   | Purpose                              |
| ---------------- | ------------------------- | ------------------------------------ |
| `EMB_MODEL`      | `nomic-embed-text`        | Embedder for chunks/sanity/verifier. |
| `REASONER_MODEL` | `llama3.1:8b`             | LLM for rerank & judge.              |
| `CHUNK_COLL`     | `trope-miner-v1-cos`      | Chroma collection for chunk vectors. |
| `TROPE_COLL`     | `trope-catalog-nomic-cos` | Chroma collection for trope vectors. |

### Retrieval / rerank / priors (judge)

| Variable                | Default | Purpose                                                    |
| ----------------------- | ------- | ---------------------------------------------------------- |
| `THRESHOLD`             | `0.25`  | **Final acceptance threshold** on **adjusted** confidence. |
| `RERANK_TOP_K`          | `8`     | Stage-1 KNN (per-work) candidates.                         |
| `RERANK_KEEP_M`         | `3`     | Stage-2 LLM keeps M support snippets.                      |
| `DOWNWEIGHT_NO_MENTION` | `0.55`  | Prior when no lexical mention & low semantic sim.          |
| `SEM_SIM_THRESHOLD`     | `0.36`  | Similarity cut for priors.                                 |
| `TROPE_TOP_K`           | `16`    | Catalog shortlist size per scene.                          |

### 2A — Semantic seeding

| Variable            | Default | Purpose                                                                   |
| ------------------- | ------- | ------------------------------------------------------------------------- |
| `SEM_TAU`           | `0.70`  | Min similarity (1 − distance) to keep a semantic seed.                    |
| `SEM_TOP_N`         | `8`     | Top-N chunks pulled per trope from Chroma (filtered to the current work). |
| `SEM_PER_SCENE_CAP` | `3`     | Cap seeds per (trope, scene) to avoid clutter.                            |

> Tuning tip: raise `SEM_TAU` (e.g., 0.75) or lower `SEM_PER_SCENE_CAP` for higher precision.

### A — Span verifier

| Variable                  | Default | Purpose                                                |
| ------------------------- | ------- | ------------------------------------------------------ |
| `SPAN_VERIFIER_THRESHOLD` | `0.25`  | Min similarity; flag spans below as `low_sim`.         |
| `SPAN_VERIFIER_MAX_SENT`  | `2`     | Expand/shrink to nearest sentence within ±N sentences. |

### B — Anti-patterns & negation/meta

**Seeding anti-suppression (gazetteer)**

| Variable      | Default | Purpose                                                                        |
| ------------- | ------- | ------------------------------------------------------------------------------ |
| `ANTI_WINDOW` | `60`    | Characters left/right of a boundary hit to look for anti-phrases / **anti-X**. |

**Verifier pass (after judge)**

| Variable          | Default      | Purpose                                  |
| ----------------- | ------------ | ---------------------------------------- |
| `NEGATION_MODE`   | `downweight` | `flag-only` \| `downweight` \| `delete`. |
| `NEG_DOWNWEIGHT`  | `0.6`        | Factor for negation cues.                |
| `META_DOWNWEIGHT` | `0.75`       | Factor for meta/parody/deconstruct cues. |
| `AA_DOWNWEIGHT`   | `0.5`        | Factor for explicit anti-aliases.        |

### C — Calibration

| Variable          | Default        | Purpose                                       |
| ----------------- | -------------- | --------------------------------------------- |
| `CALIB_BINS`      | `10`           | Reliability bins.                             |
| `CALIB_STEP`      | `0.01`         | Threshold sweep step.                         |
| `CALIB_OBJECTIVE` | `f1@precision` | `f1` \| `f1@precision` \| `precision@recall`. |
| `CALIB_MIN_PREC`  | `0.70`         | Precision floor for `f1@precision`.           |
| `CALIB_MIN_REC`   | `0.10`         | Recall floor for `precision@recall`.          |

---

## Common tweaks (copy/paste)

**Use a stricter final threshold**

```bash
THRESHOLD=0.55 ./runingest.sh
```

**Switch the reasoning model**

```bash
REASONER_MODEL=qwen2.5:7b-instruct ./runingest.sh
```

**Stronger anti-phrase suppression at seed time**

```bash
ANTI_WINDOW=100 ./runingest.sh
```

**Make the span verifier more conservative**

```bash
SPAN_VERIFIER_THRESHOLD=0.35 SPAN_VERIFIER_MAX_SENT=3 ./runingest.sh
```

**Negation/meta policy: delete instead of downweight**

```bash
NEGATION_MODE=delete ./runingest.sh
```

**Semantic seeding: tighter precision**

```bash
SEM_TAU=0.75 SEM_PER_SCENE_CAP=2 ./runingest.sh
```

**Calibrate for higher precision (after reviewing in the UI)**

```bash
CALIB_OBJECTIVE=f1@precision CALIB_MIN_PREC=0.85 ./runingest.sh
# Then use the recommended threshold printed at the end:
THRESHOLD=0.62 ./runingest.sh
```

---

## Optional analytics & viz

**Heatmap (scenes × top-20 tropes)**

```bash
python scripts/heatmap.py --db ./tropes.db --work-id <WORK_UUID> \
  --csv out/heatmap.csv --png out/heatmap.png --top 20
```

**Co-occurrence graph (CSV / GraphML / PNG)**

```bash
python scripts/cooccur.py --db ./tropes.db --work-id <WORK_UUID> \
  --threshold 0.55 \
  --out-csv out/cooccur.csv \
  --out-graphml out/cooccur.graphml \
  --png out/cooccur.png --top-n 20 --min-weight 2
```

---

## Makefile helpers (optional)

From `ingester/`, you can also use:

```bash
# Boundary seeding
gmake seed-boundary DB=./tropes.db WORK_ID=<WORK_UUID>

# Semantic seeding
gmake seed-semantic DB=./tropes.db WORK_ID=<WORK_UUID> TAU=0.72 TOP_N=10 PER_SCENE_CAP=2

# Judge only (reuse existing candidates)
gmake judge DB=./tropes.db WORK_ID=<WORK_UUID>

# Heatmap
gmake heatmap DB=./tropes.db WORK_ID=<WORK_UUID>

# Co-occurrence
gmake cooccur DB=./tropes.db WORK_ID=<WORK_UUID>
```

> If a target is missing, copy the target stanza from each script’s header into your `Makefile`.

---

## Review UI (accept/reject/edit)

```bash
cd review
export TROPES_DB=../ingester/tropes.db
python app.py
# http://127.0.0.1:5050

# API smoke:
curl -X POST http://127.0.0.1:5050/api/decision \
  -H 'Content-Type: application/json' \
  -d '{"finding_id":"<FINDING_ID>","decision":"accept"}'

curl -X POST http://127.0.0.1:5050/api/edit_span \
  -H 'Content-Type: application/json' \
  -d '{"finding_id":"<FINDING_ID>","start":123,"end":164}'
```

---

## Troubleshooting

* **“unable to open database file”** — Run from `ingester/` and ensure write permissions.
* **Chroma connection errors** — Confirm Chroma is running and `CHROMA_HOST`/`CHROMA_PORT` are correct.
* **Ollama empty embedding / model not found** — Pull or run the models (`EMB_MODEL`, `REASONER_MODEL`) and confirm `OLLAMA_BASE_URL`.
* **Semantic seeding returns zero** — Lower `SEM_TAU` (e.g., 0.65), raise `SEM_TOP_N`, or confirm `CHUNK_COLL` and per-work filtering.
* **No calibration output** — Accept/reject at least a handful of findings in the Review UI; the script detects labels automatically.

---

## Outputs

* `out/findings.csv` — machine findings after A+B adjustments
* `out/report_<WORK_ID>.html` — per-work HTML with highlighted spans (links & group colors if present)
* `out/support_<WORK_ID>.md` — support/sanity snapshot (best effort)
* `out/heatmap.csv` / `out/heatmap.png` — scene × trope heatmap (optional)
* `out/cooccur.csv` / `out/cooccur.graphml` / `out/cooccur.png` — co-occurrence graph (optional)
* `out/calibration.csv` & `out/calibration.png` — only when labels exist

---

### Tip: reproducible runs

This pipeline **resets the DB** each time. For clean comparisons:

* Keep the same `THRESHOLD`, models, and env, or
* Encode differences in filenames (e.g., `out/findings.t055.csv`), and/or
* Save the recommended threshold from calibration and reuse it on the next run.

Happy mining!
