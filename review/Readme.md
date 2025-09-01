# Trope Miner — **Review UI**

A lightweight, local Flask app to **inspect, validate, and curate** trope findings produced by the ingester. It renders scenes with inline highlights, lets you **accept / reject / edit** spans, and provides a few CLI utilities to generate trust artifacts, verify spans, and run quick calibration.

* No cloud dependencies for the UI itself; everything reads from your **SQLite** DB.
* Optional: span verification utilities use **Ollama** (embeddings) and **Chroma** (vector search).

---

## Contents

* [Quick start](#quick-start)
* [What this exposes](#what-this-exposes)
* [Directory layout](#directory-layout)
* [Configuration](#configuration)
* [Run the app](#run-the-app)
* [Make targets (gmake)](#make-targets-gmake)
* [CLI utilities](#cli-utilities)

  * [HTML highlight report](#1-html-highlight-report)
  * [Span verifier (embed-similarity + sentence snap)](#2-span-verifier-embed-similarity--sentence-snap)
  * [Calibration mini-set (P/R/F1)](#3-calibration-mini-set-prf1)
* [Front‑end behavior](#front-end-behavior)
* [API reference](#api-reference)
* [Data model notes](#data-model-notes)
* [Troubleshooting](#troubleshooting)
* [FAQ](#faq)

---

## Quick start

```bash
# from the repo root, DB lives under ingester/
cd review

# Start the UI (debug) against the ingester DB
gmake run DB=../ingester/tropes.db

# Open the app
gmake open

# Optional: warm the app once (ensures review view exists)
gmake warm
```

The server reads `TROPES_DB` (defaults to `../ingester/tropes.db`). Index → Work → Scene pages are served by `app.py`.&#x20;

---

## What this exposes

* **Index** (`/`): list works with total findings and reviewed count (based on latest human decision per finding).&#x20;
* **Work** (`/work/<work_id>`): list scenes with **chars**, **findings**, **accepted**, **rejected**.&#x20;
* **Scene** (`/scene/<scene_id>`): scene text sliced from `work.norm_text[char_start:char_end]` + finding cards and a highlight view.&#x20;
* **API**:

  * `POST /api/decision` — **accept / reject** a finding (writes an audit row).&#x20;
  * `POST /api/edit_span` — correct span and/or relabel trope (writes audit row + updates canonical).&#x20;
  * `POST /api/new_finding` — add a human finding (clamped offsets; duplicate‑safe).&#x20;
  * `GET /healthz` — liveness + DB path.&#x20;

---

## Directory layout

```
review/
├── app.py                     # Flask server + routes + API
├── Makefile                   # review Make targets (run, reports, verify, calibrate, …)
├── static/
│   ├── app.css
│   ├── review.css
│   ├── app.js
│   └── review.js             # highlighter + card interactions
├── templates/
│   ├── layout.html
│   ├── index.html
│   └── scene.html
└── scripts/
    ├── report_highlights.py   # 1) static HTML highlight reports
    ├── verify_spans.py        # 2) embed-similarity + sentence snap
    └── calibrate_mini.py      # 3) mini calibration set (P/R/F1)
```

---

## Configuration

**Database**

* `TROPES_DB`: path to SQLite DB. Defaults to `../ingester/tropes.db`.&#x20;

**Span verifier (optional)**

* `OLLAMA_BASE_URL` (e.g., `http://localhost:11434`)
* `EMBED_MODEL` (e.g., `nomic-embed-text`)
* `CHROMA_HOST`, `CHROMA_PORT` (e.g., `localhost:8000`)
* `CHUNK_COLLECTION` (e.g., `trope-miner-v1-cos`), `TROPE_COLLECTION` (e.g., `trope-catalog-nomic-cos`)

These are surfaced as variables in the `review/Makefile` targets.

---

## Run the app

```bash
# Start in debug mode, binding TROPES_DB
gmake run DB=../ingester/tropes.db

# Health check (in another shell)
gmake health

# Open browser
gmake open
```

If you plan to deep‑link directly to `/work/<id>` or `/scene/<id>` on a fresh app boot, hit `/` once (or run `gmake warm`) so the view used for “latest human decision per finding” exists. Index creates it.&#x20;

---

## Make targets (gmake)

From `review/`:

```
gmake run DB=../ingester/tropes.db          # start UI
gmake open                                  # open browser
gmake health                                # GET /healthz
gmake dbcheck DB=../ingester/tropes.db      # quick counts (works/scenes/findings)
gmake reports DB=../ingester/tropes.db TITLE="The Girl"
gmake reports DB=../ingester/tropes.db WORK_ID=<uuid>
gmake reports-all DB=../ingester/tropes.db  # all works → review/reports/*.html
gmake verify DB=../ingester/tropes.db WORK_ID=<uuid>        # dry-run span verifier
gmake verify DB=../ingester/tropes.db WORK_ID=<uuid> APPLY=1# apply suggested snaps
gmake calibrate DB=../ingester/tropes.db K=10 THRESHOLD=0.50 IOU=0.30
gmake clean                                  # remove reports/ and __pycache__
```

---

## CLI utilities

### 1) HTML highlight report

Generate a **self‑contained static HTML** per work with `<mark>` highlights and jump links. Good for sharing/review outside the app.

```bash
# by title
gmake reports DB=../ingester/tropes.db TITLE="The Girl"

# by work id
gmake reports DB=../ingester/tropes.db WORK_ID=<uuid>

# all works
gmake reports-all DB=../ingester/tropes.db
```

Output → `review/reports/<work-title>.html`.

### 2) Span verifier (embed‑similarity + sentence snap)

Dry‑run checks each finding’s span:

* **Trope alignment**: cosine similarity between span embedding and trope embedding.
* **Local coherence**: average similarity to nearest narrative chunks.
* **Sentence snap**: proposes snapping to sentence boundaries if similarity improves by Δ (default 0.05).

```bash
# dry-run
gmake verify DB=../ingester/tropes.db WORK_ID=<uuid>

# apply suggested snaps (writes audit row + updates finding)
gmake verify DB=../ingester/tropes.db WORK_ID=<uuid> APPLY=1
```

Requires Ollama + Chroma running and collections populated.

### 3) Calibration mini‑set (P/R/F1)

Pick **K scenes** (or pass explicit ids), compare current findings against **latest human accepts**, and report **precision / recall / F1** (overlap by IoU and same trope id).

```bash
# sample 10 scenes, τ=0.50
gmake calibrate DB=../ingester/tropes.db K=10 THRESHOLD=0.50 IOU=0.30
```

---

## Front‑end behavior

* **Templates**

  * `layout.html` loads base + review CSS/JS.
  * `index.html` renders either Works or Scenes table.
  * `scene.html` renders the scene text in `<pre id="text" data-offset="...">`, plus a **findings list** with absolute offsets and a **“show all highlights”** checkbox.&#x20;

* **Highlighter (`static/review.js`)**

  * Uses the scene’s **absolute base offset** to convert finding spans into scene‑relative ranges.
  * **Jump** paints a single `<mark>` and scrolls it into view; **Show all** paints all spans deterministically.
  * Reads spans from the **cards** and, if present, from the `#spans-json` payload injected by the server.&#x20;

> Note: If your text contains emoji/astral characters and you need pixel‑perfect alignment, use the emoji‑safe variant of the highlighter (code‑point → UTF‑16 code‑unit mapping). The default file clamps against the scene range and renders reliably for BMP text; swap in the “emoji‑safe” version if needed.&#x20;

---

## API reference

All endpoints are **local** (same Flask server). Examples use `jq` for pretty output.

### `POST /api/decision` — accept/reject a finding

Writes an append‑only row to `trope_finding_human` (audit trail). Decisions are one of `accept|reject`.&#x20;

```bash
curl -s http://127.0.0.1:5050/api/decision \
  -H 'Content-Type: application/json' \
  -d '{"finding_id":"<uuid>","decision":"accept","reviewer":"me"}' | jq
```

### `POST /api/edit_span` — correct offsets and/or relabel

* Clamps to valid range; requires `end > start`.
* Writes a history row (`decision='edit'`) and updates the canonical finding.&#x20;

```bash
curl -s http://127.0.0.1:5050/api/edit_span \
  -H 'Content-Type: application/json' \
  -d '{"finding_id":"<uuid>","start":123,"end":150,"trope_id":"<trope-uuid>","note":"tighten span"}' | jq
```

### `POST /api/new_finding` — add a human finding

* Adds a “human” finding with confidence + rationale; clamps/validates offsets; ignores exact duplicates.&#x20;

```bash
curl -s http://127.0.0.1:5050/api/new_finding \
  -H 'Content-Type: application/json' \
  -d '{
        "scene_id":"<scene-uuid>",
        "work_id":"<work-uuid>",
        "trope_id":"<trope-uuid>",
        "start":234,"end":260,
        "confidence":0.8,
        "rationale":"explicit wording"
      }' | jq
```

### `GET /healthz`

Returns `{ ok: true, db: "<path>" }`.&#x20;

---

## Data model notes

The UI consumes the subset below (produced by the ingester):

* **work** `(id, title, author, norm_text, …)`
* **scene** `(id, work_id, idx, char_start, char_end, …)`
* **trope** `(id, name, …)`
* **trope\_finding** `(id, work_id, scene_id, trope_id, confidence, evidence_start, evidence_end, rationale, model, …)`
* **trope\_finding\_human** *(audit)* `(finding_id, decision accept|reject|edit, corrected_start/end, corrected_trope_id, note, reviewer, created_at)`
* **v\_latest\_human** *(view)*: latest human decision per finding, created on server startup/index access.&#x20;

**Offsets:** `evidence_start/end` are **absolute character indices** into `work.norm_text`. The scene page slices the scene substring and converts absolute → scene‑relative for highlighting. &#x20;

---

## Troubleshooting

**“Scene text is blank until I click ‘Show all highlights’ or ‘Jump’.”**
This can be caused by CSS clamping/overlays. The UI ships a simple workaround: once the pane is interacted with, the text is re‑rendered with `<mark>`s and becomes visible. You can also add a small “first‑paint self‑heal” in `review.js` to auto‑render all highlights if the pane height is suspiciously small.&#x20;

**Deep‑linking directly to `/work/<id>` or `/scene/<id>` throws a view error (first boot).**
Hit `/` once (or run `gmake warm`) to ensure `v_latest_human` exists. If you prefer, call `ensure_review_schema(db)` inside those routes as well.&#x20;

**Buttons show ‘Accepted/Rejected’ but don’t persist.**
POST to `/api/decision` from the front‑end. The API routes are implemented; you may wire them in `review.js` (the sample handler is trivial). &#x20;

**Emoji / astral characters misalign highlights.**
Swap in the emoji‑safe highlighter (code‑point → UTF‑16 code‑unit conversion) if your corpus uses many emoji; the default highlighter clamps indices but doesn’t remap surrogate pairs.&#x20;

---

## FAQ

**Q: Where does the app look for the DB?**
A: `TROPES_DB` env var. Defaults to `../ingester/tropes.db`. See `DB_PATH` resolution in `app.py`.&#x20;

**Q: Does editing a span overwrite history?**
A: No. Each edit writes a row to `trope_finding_human` with `decision='edit'`. The canonical finding is updated separately so you keep a full audit trail.&#x20;

**Q: How do ‘reviewed’ counts work?**
A: Pages join through `v_latest_human` to count the **latest** decision per finding.&#x20;

**Q: Is quote normalization changing offsets?**
A: Display‑time normalization maps a few MacRoman‑style glyphs 1:1 and **does not change string length**, so offsets remain valid.&#x20;

---

### License & contributions

Internal tool for local review and dataset curation. Feel free to extend (`/api/edit_span`, `/api/new_finding` are intentionally simple), add keyboard shortcuts, or switch the highlighter to the emoji‑safe variant for maximal robustness.&#x20;


