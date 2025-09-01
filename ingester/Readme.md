# Trope Miner — **Ingestor**

The **ingester** turns a work of fiction into a **local, reproducible dataset of trope findings**:

1. **Ingest & segment** the raw text into `chapter → scene → chunk` with absolute character offsets.
2. **Embed** chunks and the **trope catalog** into Chroma (cosine space) using **Ollama** embeddings.
3. **Seed candidates** via a gazetteer (names/aliases) for fast lexical recall.
4. **Judge scenes** with a local LLM:

   * **Work‑local KNN retrieval** from Chroma,
   * **LLM re‑rank** to pick the best 2–3 support snippets,
   * **Trope sanity** (lexical + semantic) that **down‑weights** unlikely candidates before judging.
5. **Persist** findings with text‑grounded evidence spans (absolute offsets into `work.norm_text`).
6. (Optional) **Export** CSV/Markdown, generate a **support report**, and hand off to the Review UI for human curation.

> The Review UI reads the same SQLite DB (`TROPES_DB`) and provides Accept/Reject/Edit endpoints and a highlight viewer that maps absolute offsets to scene‑relative spans. &#x20;

---

## What you get

* **Evidence‑level** trope detections with:

  * `trope_id`, calibrated `confidence`,
  * `evidence_start/evidence_end` (absolute char offsets into `work.norm_text`),
  * short `rationale` + `model` used.
* **Support artifacts**:

  * `scene_support` — chosen chunk IDs + rerank notes per scene,
  * `trope_sanity` — per‑(scene,trope) sanity metrics: `lex_ok`, `sem_sim`, `weight`.
* Everything runs **fully local** (SQLite + Chroma + Ollama).

---

## Prerequisites

* **SQLite 3**
* **Ollama** running locally (e.g., `http://localhost:11434`) with:

  * embed model: `nomic-embed-text` (default),
  * reasoner model: e.g. `qwen2.5:7b-instruct` (default in Makefile) or `llama3.1:8b`.
* **Chroma** server (HTTP) at `localhost:8000` (default), with collections:

  * **Chunks**: `trope-miner-nomic-cos` (cosine),
  * **Tropes**: `trope-catalog-nomic-cos` (cosine).

---

## Quick start

From `ingester/`:

```bash
# Show targets & defaults
gmake help

# End-to-end (fresh DB -> load tropes -> seed -> judge)
gmake quickstart DB=./tropes.db WORK_ID=<WORK_UUID> MODEL="qwen2.5:7b-instruct" OLLAMA=http://localhost:11434

# Optional: export a support report (chosen snippets + sanity + findings)
gmake support-report DB=./tropes.db WORK_ID=<WORK_UUID>
```

> If you already have a populated DB (work/scene/chunk present), you can jump to **Seeding** and **Judging**.

---

## Makefile targets (most used)

* **DB**

  * `init-db` / `init-db-fresh` — apply schema (`sql/ingestion.sql`), fresh removes existing DB.
  * `backup-db` — copy `tropes.db → tropes.db.YYYYmmdd-HHMMSS.bak`.
  * `indexes`, `vacuum`, `sanity`, `validate` — maintenance & checks.

* **Tropes**

  * `load-tropes` / `reload-tropes` — upsert/reset from `trope_seed.csv`.
  * `aliases` / `dry-aliases` — expand trope aliases with an LLM (optional enrichment).

* **Embeddings**

  * `reembed` — embed **chunks** into `trope-miner-nomic-cos` (cosine).
  * `reembed-tropes` — embed **trope catalog** into `trope-catalog-nomic-cos`.

* **Candidates & Judging**

  * `seed` — gazetteer matches per chunk (fast lexical).
  * `seed-boundary` — boundary‑aware seeding script (min len, per‑trope caps).
  * `judge` — **two‑stage rerank + sanity** + scene‑level judging (see below).
  * `judge-fresh` — wipe previous findings then run `judge`.

* **Reporting**

  * `report`, `report-excerpts` — quick SQL‑level summaries.
  * `support-report` — scene‑by‑scene rerank snippets + sanity + findings (Markdown).
  * `export-findings`, `report-findings`, `report-findings-md` — export findings to CSV/MD.

* **Other**

  * `grid` — sweep multiple thresholds for sensitivity tests.
  * `chroma-sanity`, `chroma-reset-cos` — verify/recreate cosine collections.
  * `dedupe`, `dedupe-findings`, `ensure-unique-findings`, `clean` — hygiene.

Use `gmake whereis` to confirm config and script presence before long runs.

---

## Judging (what actually happens)

When you run:

```bash
gmake judge DB=./tropes.db WORK_ID=<WORK_UUID> \
  MODEL="qwen2.5:7b-instruct" OLLAMA=http://localhost:11434 \
  TOP_K=8 THRESHOLD=0.55 TROPE_TOP_K=16
```

the pipeline:

1. **Candidate shortlist**

   * Collects **gazetteer hits** for the scene; if sparse, adds top **semantic** matches from the trope catalog.

2. **Two‑stage rerank (support selection)**

   * Query Chroma (cosine) for **work‑local** top‑K chunks.
   * Send the **scene + those K snippets** to the LLM; it returns **2–3** support IDs with a short rationale.
   * Persisted to `scene_support(scene_id, support_ids, notes)`.

3. **Trope sanity (prior weights)**
   For each candidate trope:

   * **Lexical mention?** (scene or support; name/alias)
   * **Semantic affinity?** (max cosine vs. scene/support)
     If *no lexical* **and** *low semantic* (`sem_sim < SEM_SIM_THRESHOLD`), apply a prior weight of `DOWNWEIGHT_NO_MENTION` (e.g., **0.55**).
     Metrics persisted to `trope_sanity(scene_id, trope_id, lex_ok, sem_sim, weight)`.

4. **Judge prompt**
   The LLM gets the **scene text + 2–3 support snippets + candidate definitions** and sees the **PRIOR\_WEIGHTS** hint.

5. **Write findings**
   For each item returned:

   * **Adjusted confidence** = `raw_confidence × prior_weight`,
   * Keep if `adjusted ≥ THRESHOLD` and write into `trope_finding` with evidence span and rationale.

> Evidence spans are **absolute indices** into `work.norm_text`. The Review UI slices the scene and converts absolute → relative to paint highlights.&#x20;

---

## Configuration knobs

The Makefile exposes all the usual suspects:

```makefile
DB           ?= ./tropes.db
WORK_ID      ?= <uuid-of-your-work>
MODEL        ?= qwen2.5:7b-instruct        # reasoner model (Ollama)
OLLAMA       ?= http://localhost:11434
THRESHOLD    ?= 0.25
TOP_K        ?= 8
TROPE_TOP_K  ?= 16

# Two-stage rerank & sanity
export RERANK_TOP_K         ?= 8    # K from Chroma (stage 1)
export RERANK_KEEP_M        ?= 3    # keep M snippets after LLM rerank
export DOWNWEIGHT_NO_MENTION ?= 0.55
export SEM_SIM_THRESHOLD    ?= 0.36
```

Collection names (default, cosine):

* **Chunks**: `trope-miner-nomic-cos`
* **Tropes**: `trope-catalog-nomic-cos`

Change them only if you also change the embed steps (`reembed`, `reembed-tropes`) and judging arguments.

---

## Data model (key tables)

* `work(id, title, author, norm_text, …)`
* `chapter(id, work_id, idx, char_start, char_end, …)`
* `scene(id, work_id, chapter_id, idx, char_start, char_end, …)`
* `chunk(id, work_id, scene_id, idx, char_start, char_end, text, sha256, …)`
* `trope(id, name, summary, aliases JSON|legacy …)`
* `trope_candidate(work_id, scene_id, chunk_id, trope_id, start, end, source, …)`
* `trope_finding(id, work_id, scene_id, trope_id, confidence, evidence_start, evidence_end, rationale, model, …)`
* **New support/sanity** (created automatically by the rerank step):

  * `scene_support(scene_id, support_ids JSON, notes, model, k, m, created_at)`
  * `trope_sanity(scene_id, trope_id, lex_ok, sem_sim, weight, created_at)`
* (Used by Review UI)

  * `trope_finding_human(… decision='accept'|'reject'|'edit' …)`, view `v_latest_human` created on first index load.&#x20;

**Offsets:** `evidence_start/evidence_end` are **absolute** into `work.norm_text`.
The UI displays scene substrings and computes highlights relative to `scene.char_start`.&#x20;

---

## Typical flows

### Fresh database

```bash
# 0) schema + tropes
gmake init-db-fresh DB=./tropes.db
gmake reload-tropes  DB=./tropes.db CSV=./trope_seed.csv

# 1) ingest your text into work/chapter/scene/chunk (your existing script)

# 2) embed vectors
gmake reembed         DB=./tropes.db OLLAMA=http://localhost:11434
gmake reembed-tropes  DB=./tropes.db OLLAMA=http://localhost:11434

# 3) seed candidates
gmake seed-boundary   DB=./tropes.db WORK_ID=<WORK_UUID>

# 4) judge with rerank + sanity
gmake judge           DB=./tropes.db WORK_ID=<WORK_UUID> MODEL="qwen2.5:7b-instruct" THRESHOLD=0.55

# 5) inspect & export
gmake support-report  DB=./tropes.db WORK_ID=<WORK_UUID>
gmake report-findings DB=./tropes.db WORK_ID=<WORK_UUID>
```

### Using an existing database

```bash
# optional: ensure vectors exist / up to date
gmake reembed DB=./tropes.db
gmake reembed-tropes DB=./tropes.db

# seed + judge
gmake seed-boundary DB=./tropes.db WORK_ID=<WORK_UUID>
gmake judge         DB=./tropes.db WORK_ID=<WORK_UUID> TOP_K=8 THRESHOLD=0.55
```

---

## Support & sanity report

A quick audit of the new retrieval path:

```bash
# Markdown report with support snippets + sanity metrics + final findings
gmake support-report DB=./tropes.db WORK_ID=<WORK_UUID>
# → out/support.<timestamp>.md
```

---

## Integration with the Review UI

Once you’ve run `judge`, switch to the `review/` sub‑project:

```bash
cd ../review
gmake run DB=../ingester/tropes.db   # serves the web app
```

* **Index → Work → Scene**: browse findings; highlights are computed from absolute offsets.&#x20;
* The server reads `TROPES_DB` and exposes `/api/decision`, `/api/edit_span`, `/api/new_finding` for human curation.&#x20;

---

## Troubleshooting

* **No retrieval hits / wrong collection:** verify Chroma collections use **cosine** and names match (`chroma-sanity`, `chroma-reset-cos`).
* **Model errors:** confirm Ollama is reachable (`OLLAMA` URL) and the requested models are pulled.
* **Unicode offsets:** spans are absolute indices into `norm_text` and survive display‑time quote fixes; the UI handles the scene‑relative mapping when painting `<mark>`s.&#x20;
* **Duplicate rows:** use `dedupe`, `dedupe-findings`, or add the `ensure-unique-findings` index.

---

## Reproducibility

* Every step is **idempotent** or guarded by uniqueness constraints and indexes.
* Findings keep `model` and `confidence` along with evidence; support and sanity are timestamped per scene.
* Exports (`export-findings`, `report-findings-md`) are generated with timestamps for audit trails.

---

## Glossary

* **Gazetteer candidate**: a weak hit (name/alias) stored in `trope_candidate`.
* **Support snippet**: a chunk chosen by LLM rerank as directly relevant evidence.
* **Sanity weight**: a prior (≤ 1.0) applied before judging to suppress candidates with no lexical mention and low semantic affinity.
* **Adjusted confidence**: `raw_confidence × sanity_weight`.

---

## See also

* **Review UI (Flask)** — DB path, routes & human decisions (**`TROPES_DB`**, `/api/*`, `v_latest_human`).&#x20;
* **Front‑end highlighter** — converts absolute offsets to scene‑relative `<mark>`s, supports **Jump** and **Show all highlights**.&#x20;

---

**Happy mining.**
