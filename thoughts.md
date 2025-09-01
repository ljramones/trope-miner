# Full Directions 

# 1) Accuracy & trust (highest ROI)

* **Better evidence picking.** After the LLM proposes a span, verify it:

  * Compute sim(span, trope-definition) and sim(span, scene) with the same embedder; drop/flag spans below a threshold.
  * If low, auto-shrink/expand to the nearest sentence boundary that maximizes similarity.
* **Negation & anti-patterns.** Add a light “verifier” pass that rejects hits with cues like “not X”, “anti-X”, or meta commentary (“this isn’t a whodunit”). Store a short `verifier_flag` in `trope_finding`.
* **Confidence calibration.** Create a tiny gold set (10–20 scenes) and:

  * Tune `--threshold`, and make a reliability plot (confidence bins vs. precision).
  * Store `calibration_version` in findings so you know which run used which threshold.

# 2) Candidate seeding that catches more while staying precise

* **Fuzzy phrase seeding.** In addition to boundary regex, add one approximate pass:

  * Embed each trope name+2–3 aliases; search chunks with a cosine cutoff (fast top-k per chunk window). Anything above τ becomes a candidate row with `source='semantic'`.
* **Anti-aliases.** Keep a small JSON list per trope of “do not fire on” phrases (e.g., “dream-like prose” shouldn’t trigger “dream-sequence”). Merge that into `seed_candidates_boundary.py`.

# 3) Retrieval & scoring upgrades

* **Work-local retrieval (you added this—great).** Next: **two-stage rerank**:

  1. Chroma gets top-k chunks.
  2. Send those k snippets + scene to the LLM for a *short* rerank rationale; pick top 2–3 as support. (Cuts weird far-context support.)
* **Trope shortlist sanity.** If a candidate trope is *never* mentioned lexically in the scene nor nearest chunks (no alias, no high-sim phrase), down-weight it before judging.

# 4) Outputs that are easy to read (and debug)

* **HTML report with highlights.** Generate a single HTML per work:

  * Chapter/scene list on the left; main text on the right with `<mark>` around evidence spans (color by trope).
  * Hover → shows trope name + confidence + rationale.
* **Chapter/scene heatmap.** Simple CSV/PNG: rows=scenes, cols=top 20 tropes, cells=confidence (or count). Makes patterns pop.
* **Rationales next to spans.** In CSV/MD exports, add columns: `scene_idx`, `chapter_idx`, `evidence_sentence`.

# 5) Human-in-the-loop loop (fast curation)

* **Review queue.** A tiny CLI or web page that shows (scene text + proposed spans) with \[Accept / Reject / Edit span]. Persist decisions into `trope_finding_human`.
* **Active learning.** Train a tiny logistic model (or LLM prompt adjustment) on accepted vs. rejected to auto-tune thresholds per trope.

# 6) Ontology & coverage

* **Trope families.** Add `trope_group(id, name)` and `trope_group_member(trope_id, group_id)`. Let the report cluster results (“Mystery”, “Horror”, “Romance”).
* **Link out.** Keep `tvtropes_url` or `wikidata_qid`. Export clickable links in the HTML report.
* **Cross-trope co-occurrence.** Compute a graph per work; show “Detective ↔ Whodunit” edges with strength.

# 7) Robustness & scale

* **Per-work collections (optional mode).** Toggle to create a fresh chunk collection per work (no leakage, easy cleanup). Keep your current “global + where filter” as default.
* **Stamp runs.** Add a `run_id` table with parameters (models, versions, thresholds). Attach `run_id` to all findings—repro is then trivial.
* **Batching many works.** Add a `find_new_works.sh` that ingests everything under a folder and writes one report per work.

# 8) Developer polish / DX

* **`make demo`**: create a tiny “toy” novella and run end-to-end in \~30s.
* **Unit tests for regex builder and span math.** Catch off-by-ones early.
* **Telemetry for failure modes.** Count how often each candidate source leads to accepted findings.

---

Totally. Here’s a tight, practical plan to knock down the **remaining items**, grouped by ROI and exactly where to wire them in. I’ll point to the files you already have and spell out schema tweaks + quick algorithms. If you want full files for any piece, say the word and I’ll drop them in.

# 1) Accuracy & trust

## A. Better evidence picking (span verification & auto-snap)

**Where:** extend `ingester/scripts/span_verifier.py`.

**Schema:** no change.

**Algo (fast + local):**

1. For each finding:

   * Pull `span = norm_text[evidence_start:evidence_end]`, the scene text, and the **trope definition**: `name + summary` from `trope`.
2. Compute `sim(span, trope_def)` and `sim(span, scene)` via your Ollama embedder (same model as elsewhere).
3. If either < τ (e.g., 0.32), try a **boundary-aware search** in the scene:

   * Enumerate candidate windows from sentence boundaries around the original span (e.g., ±1–2 sentences, capped to ≤ 280 chars).
   * Score each window by `α*sim(win,trope_def) + (1-α)*sim(win,scene)` (α≈0.7).
   * Pick the best; if it beats the original by δ (e.g., +0.05), **replace** `evidence_start/end`.
4. Persist updated spans in place (exactly like your snapper does).

**Make:** keep `span-snap-apply` for pure snapping; add `span-verify` later to run this semantic pass.

---

## B. Negation & anti-patterns (light verifier pass)

**Where:** new `ingester/scripts/verifier_pass.py`, called after judging.

**Schema:** add a nullable flag to findings:

```sql
ALTER TABLE trope_finding ADD COLUMN verifier_flag TEXT;  -- e.g., 'negation', 'meta', 'antialias'
```

**Heuristics (cheap & effective):**

* **Negation around alias:** within ±40 chars of alias mention: `\b(no|not|never|without|isn’t|isn't|wasn’t|wasn't|anti-...)\b`.
* **Meta disqualifiers:** “this isn’t a whodunit”, “deconstructs X”, “parody of X”, “subverts X”.
* **Anti-aliases (see §2B):** if any “do-not-fire” phrase co-occurs, flag.

**Policy:** if a hit is flagged:

* Either **drop** it (delete row) or **down-weight**: `confidence *= 0.6` and set `verifier_flag`.
* I suggest **do not delete automatically**; down-weight + flag → the UI still surfaces it for review.

---

## C. Confidence calibration (versioned)

**Where:** you already have `calibrate_threshold.py`. Add a `--calibration-version` and write it onto findings of the *next* judge run.

**Schema:**

```sql
ALTER TABLE trope_finding ADD COLUMN calibration_version TEXT;
```

**Run flow:**

1. Label your small gold (10–20 scenes) in the UI.
2. `gmake calibrate-plot` → pick THRESHOLD (e.g., 0.42).
3. When running `judge-scenes`, pass `--threshold 0.42` and set env `CALIBRATION_VERSION=v1`. In `trope_miner_tools.py`, add:

   * Read `os.getenv("CALIBRATION_VERSION")`; write to `trope_finding.calibration_version` on insert.

---

# 2) Candidate seeding (recall without chaos)

## A. **Semantic seeding** (very high ROI)

**Where:** new `ingester/scripts/seed_candidates_semantic.py`.

**Approach:**

* For each trope: embed `name + ". " + summary` (or top 2–3 aliases).
* Query **chunk** collection with `where={"work_id": WORK_ID}`, `n_results=topN` (start with 5–10).
* For each returned chunk with `similarity ≥ τ_sem` (start 0.7 sim, i.e., `1-distance ≥ 0.7`):

  * Insert `trope_candidate` with `source='semantic'`, `score=similarity`.

**Keep precision:** union with boundary seeding, but **do not** explode: cap per-trope per-scene (e.g., ≤ 3 semantic seeds).

**Make:** add `seed-semantic` target.

---

## B. **Anti-aliases**

**Where:** extend your CSV + loader or add a tiny table.

**Schema option 1 (simple):** add JSON column to `trope`:

```sql
ALTER TABLE trope ADD COLUMN anti_aliases TEXT;  -- JSON array of phrases
```

Parse it in `seed_candidates_boundary.py`; if an anti-alias regex hits the same chunk/scene, **skip** the candidate. Reuse your `build_pattern()` but without pluralization.

---

# 3) Retrieval & scoring (you’ve got the big ones)

You already shipped:

* Work-local retrieval ✅
* Two-stage rerank ✅
* Sanity priors ✅

Optional small boost:

* In LLM rerank, add a “**penalize generic background**” hint (you already do).
* Pass the **stage-1 similarity** into the LLM as side info (“KNN score: 0.83”) to help it de-prefer weak ones.

---

# 4) Outputs (quick wins)

## A. **HTML report** (you already have): `scripts/report_html.py`

Enhance:

* Link trope names to `tvtropes_url` (see §6B).
* Color by **trope group** (see §6A).

## B. **Chapter/scene heatmap**

**Where:** new `ingester/scripts/heatmap.py`.

**CSV:** rows=scenes, cols=top 20 tropes by frequency; values = max confidence per scene.
**PNG:** simple matplotlib `imshow`; save `out/heatmap.png`.

**Make:** `gmake heatmap`.

## C. **Rationales next to spans in exports**

**Where:** `scripts/export_findings.py`

* Add columns: `scene_idx`, `chapter_idx`, `evidence_sentence` (derive by snapping the span to nearest sentence).
* Already trivial to compute using your snap logic.

---

# 5) Human-in-the-loop

## A. Review queue ✅ (UI exists)

## B. **Active learning**

**Where:** new `ingester/scripts/learn_thresholds.py`.

**Plan:**

* Train a tiny logistic **per trope**: features = `[raw_conf, prior, (raw_conf*prior)]`.
* Fit to human labels (accept=1, reject=0).
* Emit `trope_thresholds` table: `(trope_id TEXT PRIMARY KEY, threshold REAL)`.
* Modify `judge_scenes` to load a per-trope threshold if present; else default.

---

# 6) Ontology & coverage

## A. **Trope families**

**Schema:**

```sql
CREATE TABLE IF NOT EXISTS trope_group(
  id   TEXT PRIMARY KEY,
  name TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS trope_group_member(
  trope_id TEXT NOT NULL,
  group_id TEXT NOT NULL,
  PRIMARY KEY (trope_id, group_id)
);
```

Use in HTML report to group sections and color-code highlights.

## B. **Link out**

**Schema:**

```sql
ALTER TABLE trope ADD COLUMN tvtropes_url TEXT;
ALTER TABLE trope ADD COLUMN wikidata_qid TEXT;
```

Populate via CSV; show as `<a>` in HTML and add to MD export.

## C. **Co-occurrence**

**Where:** new `ingester/scripts/cooccur.py`

* Build a per-scene set of present tropes (adj\_conf ≥ THRESHOLD); increment edge counts for pairs.
* Export CSV or simple GraphML; optional PNG chord diagram.

---

# 7) Robustness & scale

## A. **Per-work collections (toggle)**

**Code touch:** `embedder.py`, `rerank_support.py`

* If `PER_WORK_COLLECTIONS=1`, name collections as `f"{CHUNK_COLL}__{work_id}"`.
* On retrieval, open that specific collection; default remains global + `where={"work_id": ...}`.

## B. **Stamp runs (full repro)**

**Schema:**

```sql
CREATE TABLE IF NOT EXISTS run(
  id TEXT PRIMARY KEY,
  created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
  params_json TEXT  -- models, thresholds, seeds, knobs
);
ALTER TABLE trope_finding ADD COLUMN run_id TEXT;  -- nullable for back-compat
```

**Code:** in `judge_scenes`, create a run row at start, stash CLI/env into JSON, and write `run_id` on every finding.
Add `gmake runs` to list past runs.

## C. **Batching many works**

**Where:** new `ingester/scripts/batch_ingest.py`

* Walk a folder of texts, call your existing `ingestor_segmenter.py` + `judge-scenes` + `report_html.py` per file.
* Dump one report per `work_id`.

---

# 8) DX polish

## A. `make demo`

* Add a tiny toy novella under `demo/`; single Make target that:

  * creates a fresh DB, loads a trimmed trope CSV, ingests demo text, runs judge, opens report.

## B. Unit tests

* `tests/test_regex_builder.py` for `build_pattern` (dash/whitespace/plural).
* `tests/test_span_math.py` for snap boundaries & clamping.
* Wire into `gmake test-smoke` or add `gmake test-all`.

## C. Telemetry

* Add a view or script that joins:

  * `trope_candidate.source` → (`gazetteer`/`semantic`)
  * final acceptance (`v_latest_human.decision='accept'`)
* Output per‐source precision and contribution.

---

## Suggested execution order (fastest impact first)

1. **Span verification** (1A) ➜ **Negation/anti-patterns** (1B).
2. **Semantic seeding** (2A) + **Anti-aliases** (2B).
3. **Calibration versioning** (1C) and optionally **per-trope thresholds** (5B).
4. **Heatmap + export enrichments** (4B/4C).
5. **Ontology + links** (6A/6B).
6. **Run stamping** (7B) and, if needed, **per-work collections** (7A).
7. **Batching** + **DX** (7C/8).


