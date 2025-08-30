# Trope Miner — Setup

This guide bootstraps the SQLite database, Chroma collections, and local LLM/embedding stack used by **Trope Miner**.

---

## 🚀 Quickstart (for the impatient)

```bash
# 1. Initialize schema + reload trope catalog from CSV
gmake init-db reload-tropes

# 2. Run full pipeline: expand aliases → seed boundary-aware candidates → report
gmake all-boundary

# 3. Judge with LLM
gmake judge
```

That’s it. The rest of this doc explains each piece in detail.

---

## 0. Prerequisites

* **Python** 3.10+

* **SQLite CLI** (`sqlite3`)

* **Pip packages**: `chromadb`, `requests`, `tqdm`

  ```bash
  pip install chromadb requests tqdm
  ```

* **Ollama** running locally
  (Mac: `brew install ollama && brew services start ollama`)

  Pull the models you plan to use:

  ```bash
  ollama pull nomic-embed-text
  ollama pull qwen2.5:7b-instruct    # good JSON adherence
  ollama pull llama3.1:8b            # optional reasoner
  ollama pull mxbai-embed-large      # optional embedder
  ```

  Quick API sanity check (embedding call works):

  ```bash
  curl -s http://localhost:11434/api/embeddings \
    -d '{"model":"nomic-embed-text","prompt":"hello"}' | jq '.embedding | length'
  ```

* **Chroma** vector DB server reachable at `localhost:8000`
  Easiest: Docker

  ```bash
  docker run -d --name chroma -p 8000:8000 ghcr.io/chroma-core/chroma:latest
  ```

* **Modern GNU Make** (`gmake`) — macOS ships `make 3.81` which is too old.
  Install via Homebrew:

  ```bash
  brew install make
  alias make=gmake   # optional convenience
  ```

---

## 1. Initialize the SQLite schema

```bash
gmake init-db
```

This runs `ingestion.sql` into `tropes.db` and adds helpful indexes.

---

## 2. Load the trope catalog

Seed the `trope` table from `trope_seed.csv`:

```bash
# Upsert rows
gmake load-tropes

# Wipe table then reload from CSV
gmake reload-tropes
```

Sanity checks:

```bash
sqlite3 ./tropes.db "SELECT COUNT(*) FROM trope;"
sqlite3 ./tropes.db "SELECT id,name FROM trope ORDER BY name LIMIT 10;"
```

---

## 3. Ingest and segment text

```bash
python ingestor_segmenter.py ingest \
  --db ./tropes.db \
  --file ../data/TheGirl.txt \
  --title "The Girl" --author "L. Mitchell" \
  --target 450 --overlap 80 --min-tokens 300 --max-tokens 600 --encoding auto
```

Check counts:

```bash
sqlite3 ./tropes.db "SELECT COUNT(*) FROM chunk;"
sqlite3 ./tropes.db "SELECT id, char_start, char_end FROM scene ORDER BY idx LIMIT 3;"
```

---

## 4. Prepare Chroma collections (cosine distance)

```python
# chroma_cosine_bootstrap.py
import chromadb
c = chromadb.HttpClient(host="localhost", port=8000)
name = "trope-miner-nomic-cos"
try: c.delete_collection(name)
except: pass
c.create_collection(name, metadata={"hnsw:space":"cosine"})
print("OK:", name)
```

Run once:

```bash
python chroma_cosine_bootstrap.py
```

---

## 5. Embed narrative chunks

```bash
export OLLAMA_BASE_URL=http://localhost:11434

python embedder.py \
  --db ./tropes.db \
  --collection trope-miner-nomic-cos \
  --model nomic-embed-text \
  --chroma-host localhost --chroma-port 8000
```

---

## 6. Embed the trope catalog

```bash
python embed_tropes.py
```

This upserts into `trope-catalog-nomic-cos`.

---

## 7. Seed candidates

Choose one:

```bash
# naive substring seeding
gmake seed

# boundary-aware regex seeding (recommended)
gmake seed-boundary
```

Check:

```bash
sqlite3 ./tropes.db "SELECT COUNT(*) FROM trope_candidate WHERE work_id='<WORK_ID>';"
```

---

## 8. Judge with LLM

```bash
gmake judge
```

Results go into `trope_finding`. Inspect:

```bash
gmake report
gmake report-excerpts
```

---

## 9. Maintenance

* Wipe candidates/findings:

  ```bash
  gmake clean
  ```
* Validate alias JSON:

  ```bash
  gmake validate
  ```
* Add indexes:

  ```bash
  gmake indexes
  ```
* Optimize DB:

  ```bash
  gmake vacuum
  ```

---

## 10. Handy queries

```bash
sqlite3 ./tropes.db "SELECT id,title,author FROM work;"
sqlite3 ./tropes.db "SELECT COUNT(*) FROM chunk;"
sqlite3 ./tropes.db "SELECT name, COUNT(*) FROM trope_candidate GROUP BY name ORDER BY 2 DESC LIMIT 10;"
```

---

## 11. Re-running safely

* Embedding is idempotent.
* To re-judge from scratch:

  ```bash
  sqlite3 ./tropes.db "DELETE FROM trope_finding WHERE work_id='<WORK_ID>';"
  gmake judge
  ```

---

👉 Everyday workflow:

```bash
gmake init-db reload-tropes
gmake all-boundary
gmake judge
```

