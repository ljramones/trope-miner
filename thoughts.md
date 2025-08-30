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

# 8) Multilingual (if you want it)

* Detect language on ingest (`lang` on `work`). If not English:

  * Use a multilingual embedder (BGE-M3-style) for seeding & retrieval.
  * Option A: judge in original language; Option B: translate scene chunk → English for the judge, but map spans back (store both indices).

# 9) Developer polish / DX

* **`make demo`**: create a tiny “toy” novella and run end-to-end in \~30s.
* **Unit tests for regex builder and span math.** Catch off-by-ones early.
* **Telemetry for failure modes.** Count how often each candidate source leads to accepted findings.

---

## Quick wins for the next day

1. **HTML highlighting report** (1 file per work). Big payoff in trust.
2. **Span verifier** (embed-similarity check + sentence boundary snap).
3. **Calibration mini-set** (hand-label 10 scenes; tune threshold and report P/R).

If you want, I can sketch:

* a minimal Flask app for the review queue,
* a `report.html` generator that reads your DB and paints spans,
* or the 20-line “span verifier” that re-scores and tightens LLM spans.
