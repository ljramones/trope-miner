# Trope Miner

*Local-first trope mining for fiction.*
Trope Miner ingests long-form text, finds narrative tropes with evidence spans, and gives you a review UI to accept/reject/tighten results. Everything runs on your machine using **SQLite**, **Chroma** (vector DB), and **Ollama** (LLMs/embeddings).

* ⚙️ **Ingest & judge**: boundary + semantic seeding → retrieval → LLM rerank → priors → judgments
* 🧪 **Human-in-the-loop**: accept/reject/edit spans; calibration; per-trope thresholds
* 🔍 **Local-first**: no cloud calls; reproducible runs w/ stamped params
* 📊 **Outputs**: HTML highlight report, CSV, heatmap, co-occurrence graph

> Detailed, hands-on runbooks live in each subproject:
> • **Ingestor**: [`ingester/RUNME.md`](ingester/Readme.md)
> • **Review UI**: [`review/README.md`](review/Readme.md)

---

## Quick start

### 1) Ingest + judge a single work

```bash
# From repo root
cd ingester
pip install -r requirements.txt

# Services (make sure they’re running)
# - Ollama: http://localhost:11434 (pull the models you plan to use)
ollama pull nomic-embed-text
ollama pull llama3.1:8b   # or qwen2.5:7b-instruct

# - Chroma server: http://localhost:8000
#   (e.g., docker run -p 8000:8000 chromadb/chroma)

# One-shot pipeline (resets DB, runs seeding + judge + reports)
./runingest.sh
```

This writes:

* `ingester/out/findings.csv`
* `ingester/out/report_<WORK_ID>.html`
* (optional) `support_*.md`, heatmap/co-occur if enabled

### 2) Review / curate

```bash
cd ../review
export TROPES_DB=../ingester/tropes.db
python app.py
# → http://127.0.0.1:5050
```

Use the UI to **accept / reject / edit**. Subsequent runs can **calibrate** thresholds from your decisions.

---

## Features

* **Two-stage retrieval**: work-local KNN (Chroma) → LLM rerank (anti-generic, KNN score side-info)
* **Sanity priors**: lexical mention + semantic affinity gates weak candidates
* **Negation / anti-patterns**: anti-aliases at seed time; negation/meta/anti-\* verifier after judge
* **Span verification**: embedding similarity + sentence snapping
* **Semantic seeding (2A)**: raise recall with a similarity cap and per-scene cap
* **Per-trope thresholds**: optional logistic fit from human labels
* **Run stamping**: every judge run is stored with its exact params for reproducibility
* **Per-work collections (toggle)**: isolate vectors per work when desired
* **Analytics**: scene×trope heatmap; co-occurrence graph (CSV/GraphML/PNG)
* **HTML report**: highlights with hover rationales; links to TVTropes; color by **trope group** if defined

---

## Repository layout

```
.
├── ingester/                # mining pipeline
│   ├── runingest.sh         # one-shot pipeline
│   ├── RUNME.md             # detailed ingest docs (env vars, knobs)
│   ├── embedder.py          # chunk → Chroma (per-work collections optional)
│   ├── rerank_support.py    # KNN → LLM rerank + sanity priors
│   ├── trope_miner_tools.py # judge-scenes entrypoint
│   ├── scripts/             # load/report/verify/seed/analytics/etc.
│   └── tropes_data/         # trope catalog CSV
└── review/                  # Flask review UI
    ├── app.py               # server + API
    ├── README.md            # review docs
    └── templates/static     # UI
```

---

## Requirements

* **Python 3.10+**
* **Ollama** running locally (for embeddings + reasoning)
* **Chroma** server (HTTP, default `localhost:8000`)
* `pip install -r ingester/requirements.txt`
* `pip install -r review/requirements.txt` (for the UI)

---

## Configuration (high-value toggles)

Most knobs are environment variables read by `runingest.sh`. See the full table in [`ingester/RUNME.md`](ingester/RUNME.md). Highlights:

* **Models & collections**

  * `EMB_MODEL=nomic-embed-text`
  * `REASONER_MODEL=llama3.1:8b` (or `qwen2.5:7b-instruct`)
  * `CHUNK_COLL=trope-miner-v1-cos`, `TROPE_COLL=trope-catalog-nomic-cos`
  * `PER_WORK_COLLECTIONS=1` → use per-work chunk collections (`<CHUNK_COLL>__<WORK_ID>`)

* **Judge**

  * `THRESHOLD=0.25` (on adjusted confidence)
  * `RERANK_TOP_K=8`, `RERANK_KEEP_M=3`
  * `DOWNWEIGHT_NO_MENTION=0.55`, `SEM_SIM_THRESHOLD=0.36`

* **Semantic seeding (2A)**

  * `SEM_TAU=0.70`, `SEM_TOP_N=8`, `SEM_PER_SCENE_CAP=3`

* **Verifier / anti-patterns**

  * `SPAN_VERIFIER_THRESHOLD=0.25`, `SPAN_VERIFIER_MAX_SENT=2`
  * `NEGATION_MODE=downweight` (`flag-only|downweight|delete`)
  * `ANTI_WINDOW=60`

* **Active learning**

  * `LEARN_THRESHOLDS=1` to fit per-trope thresholds (if you’ve labeled in the UI)

---

## Outputs

* `findings.csv` — machine findings (after A+B adjustments)
* `report_<WORK_ID>.html` — highlight report (links & group colors if present)
* `support_<WORK_ID>.md` — chosen supports + priors (best effort)
* `heatmap_*.{csv,png}` — scenes × top-20 tropes (optional)
* `cooccur_*.{csv,graphml,png}` — trope co-occurrence (optional)
* `calibration.{csv,png}` — recommended threshold (when labels exist)
* `run` table — parameters stamped for each judge run (`run_id` also written to findings)

---

## Workflow tips

* **Reproducibility**: every judge run is stamped in the `run` table with its params JSON; `trope_finding.run_id` links outputs to a run.
* **Per-work vectors**: set `PER_WORK_COLLECTIONS=1` for isolation when indexing many books.
* **Trope groups**: if you populate `trope_group`/`trope_group_member`, the HTML report colors by group and shows group badges in the legend.
* **Calibration**: after a few accepts/rejects in the UI, rerun the pipeline to get a recommended `THRESHOLD`.

---

## Batching many works

There’s a helper script to walk a folder of `.txt` and process each:

```bash
cd ingester
python scripts/batch_ingest.py \
  --root ../data/corpus \
  --pattern "*.txt" \
  --title-from "filename" \
  --author "Unknown"
```

Each work gets its own HTML report under `ingester/out/`.

---

## Contributing / extending

* Trope catalogs: edit `ingester/tropes_data/trope_seed.csv` (supports aliases, URLs, groups).
* Anti-aliases: add `anti_aliases` JSON per trope; seed-time suppression respects nearby anti-phrases and `anti-<alias>` patterns.
* Add visualizations: heatmaps and chord/co-occurrence diagrams are pure Python (Matplotlib/NetworkX); easy to extend.
* Model swaps: change `REASONER_MODEL` and `EMB_MODEL` env vars; the pipeline is model-agnostic as long as Ollama serves them.

---

## License

ASLv2.0

---

### Acknowledgements

Built around local building blocks: **SQLite**, **Chroma**, and **Ollama**. Designed for iterative, human-curated narrative analysis.
