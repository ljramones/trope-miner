-- sql/ingestion.sql (unified schema)
PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;

-- ===== Core text model =====
CREATE TABLE IF NOT EXISTS work (
    id         TEXT PRIMARY KEY,
    title      TEXT,
    author     TEXT,
    source     TEXT,
    license    TEXT,
    raw_text   BLOB,
    norm_text  TEXT,
    char_count INTEGER,
    created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
CREATE INDEX IF NOT EXISTS idx_work_title  ON work(title);
CREATE INDEX IF NOT EXISTS idx_work_author ON work(author);

CREATE TABLE IF NOT EXISTS chapter (
    id         TEXT PRIMARY KEY,
    work_id    TEXT    NOT NULL,
    idx        INTEGER NOT NULL,
    title      TEXT,
    char_start INTEGER,
    char_end   INTEGER,
    FOREIGN KEY(work_id) REFERENCES work(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_chapter_work_idx ON chapter(work_id, idx);

CREATE TABLE IF NOT EXISTS scene (
    id         TEXT PRIMARY KEY,
    work_id    TEXT    NOT NULL,
    chapter_id TEXT,
    idx        INTEGER NOT NULL,
    char_start INTEGER,
    char_end   INTEGER,
    heading    TEXT,
    FOREIGN KEY(work_id)    REFERENCES work(id)    ON DELETE CASCADE,
    FOREIGN KEY(chapter_id) REFERENCES chapter(id) ON DELETE SET NULL
);
CREATE INDEX IF NOT EXISTS idx_scene_work_idx ON scene(work_id, idx);

CREATE TABLE IF NOT EXISTS chunk (
    id          TEXT PRIMARY KEY,
    work_id     TEXT    NOT NULL,
    scene_id    TEXT,
    idx         INTEGER NOT NULL,
    char_start  INTEGER,
    char_end    INTEGER,
    token_start INTEGER,
    token_end   INTEGER,
    text        TEXT    NOT NULL,
    sha256      TEXT    NOT NULL,
    FOREIGN KEY(work_id)  REFERENCES work(id)  ON DELETE CASCADE,
    FOREIGN KEY(scene_id) REFERENCES scene(id) ON DELETE SET NULL
);
-- Replace global-unique sha with per-work uniqueness
DROP INDEX IF EXISTS idx_chunk_sha256;
CREATE UNIQUE INDEX IF NOT EXISTS idx_chunk_work_sha   ON chunk(work_id, sha256);
CREATE INDEX        IF NOT EXISTS idx_chunk_work_idx   ON chunk(work_id, idx);
CREATE INDEX        IF NOT EXISTS idx_chunk_work_scene ON chunk(work_id, scene_id, idx);
CREATE INDEX        IF NOT EXISTS idx_chunk_scene      ON chunk(scene_id);
CREATE INDEX        IF NOT EXISTS idx_chunk_work_span  ON chunk(work_id, char_start, char_end);

-- Backref to vector store (vectors live in Chroma)
CREATE TABLE IF NOT EXISTS embedding_ref (
    chunk_id   TEXT    NOT NULL,
    collection TEXT    NOT NULL,
    model      TEXT    NOT NULL,
    dim        INTEGER NOT NULL,
    chroma_id  TEXT    NOT NULL,
    PRIMARY KEY(chunk_id, collection),
    FOREIGN KEY(chunk_id) REFERENCES chunk(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_embedding_model      ON embedding_ref(model);
CREATE INDEX IF NOT EXISTS idx_embedding_collection ON embedding_ref(collection);

-- Lexical FTS mirror of chunk.text
CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(
    text,
    content='chunk',
    content_rowid='rowid'
);

-- ===== Trope catalog & mining =====
CREATE TABLE IF NOT EXISTS trope (
    id         TEXT PRIMARY KEY,
    name       TEXT NOT NULL UNIQUE,
    summary    TEXT,
    long_desc  TEXT,                        -- optional longer description
    tags       TEXT,                        -- JSON array, e.g. ["mystery","structure"]
    source_url TEXT,
    aliases    TEXT,                        -- JSON array of strings
    updated_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
CREATE INDEX IF NOT EXISTS idx_trope_name ON trope(name);

CREATE TABLE IF NOT EXISTS trope_candidate (
    id         TEXT PRIMARY KEY,
    work_id    TEXT NOT NULL,
    scene_id   TEXT,
    chunk_id   TEXT,
    trope_id   TEXT NOT NULL,
    surface    TEXT,
    alias      TEXT,
    start      INTEGER,
    end        INTEGER,
    source     TEXT DEFAULT 'gazetteer',
    score      REAL  DEFAULT 0.0,
    created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    FOREIGN KEY(work_id)  REFERENCES work(id)  ON DELETE CASCADE,
    FOREIGN KEY(scene_id) REFERENCES scene(id) ON DELETE SET NULL,
    FOREIGN KEY(chunk_id) REFERENCES chunk(id) ON DELETE SET NULL,
    FOREIGN KEY(trope_id) REFERENCES trope(id) ON DELETE CASCADE,
    CHECK (end >= start)                     -- basic span sanity
);
CREATE INDEX IF NOT EXISTS idx_tc_work_scene ON trope_candidate(work_id, scene_id);
CREATE INDEX IF NOT EXISTS idx_tc_chunk      ON trope_candidate(chunk_id);
CREATE INDEX IF NOT EXISTS idx_tc_trope      ON trope_candidate(trope_id);

CREATE TABLE IF NOT EXISTS trope_finding (
    id             TEXT PRIMARY KEY,
    work_id        TEXT NOT NULL,
    scene_id       TEXT,
    chunk_id       TEXT,
    trope_id       TEXT NOT NULL,
    level          TEXT CHECK(level IN ('span','scene','work')),
    confidence     REAL NOT NULL,
    rationale      TEXT,
    evidence_start INTEGER,
    evidence_end   INTEGER,
    created_at     TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    model          TEXT,
    FOREIGN KEY(work_id)  REFERENCES work(id)  ON DELETE CASCADE,
    FOREIGN KEY(scene_id) REFERENCES scene(id) ON DELETE SET NULL,
    FOREIGN KEY(chunk_id) REFERENCES chunk(id) ON DELETE SET NULL,
    FOREIGN KEY(trope_id) REFERENCES trope(id) ON DELETE CASCADE,
    CHECK (evidence_end IS NULL OR evidence_start IS NULL OR evidence_end >= evidence_start)
);
CREATE INDEX IF NOT EXISTS idx_tf_work_scene   ON trope_finding(work_id, scene_id);
CREATE INDEX IF NOT EXISTS idx_tf_trope        ON trope_finding(trope_id);
CREATE INDEX IF NOT EXISTS idx_finding_work    ON trope_finding(work_id);
CREATE INDEX IF NOT EXISTS idx_tf_trope_created ON trope_finding(trope_id, created_at);

CREATE TABLE IF NOT EXISTS trope_relation (
  src_id TEXT NOT NULL,
  dst_id TEXT NOT NULL,
  rel    TEXT NOT NULL
          CHECK (rel IN ('subtrope-of','parent-of','see-also','overlaps','contrast','anti')),
  PRIMARY KEY (src_id, dst_id, rel),
  FOREIGN KEY (src_id) REFERENCES trope(id) ON DELETE CASCADE,
  FOREIGN KEY (dst_id) REFERENCES trope(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_tr_src ON trope_relation(src_id, rel);
CREATE INDEX IF NOT EXISTS idx_tr_dst ON trope_relation(dst_id, rel);

CREATE TABLE IF NOT EXISTS trope_example (
  id          TEXT PRIMARY KEY,
  trope_id    TEXT NOT NULL,
  quote       TEXT,         -- short quote or description
  work_title  TEXT,         -- where it came from
  work_author TEXT,
  location    TEXT,         -- chapter/page/timecode
  url         TEXT,         -- optional external link
  FOREIGN KEY (trope_id) REFERENCES trope(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_te_trope ON trope_example(trope_id);

CREATE TABLE IF NOT EXISTS trope_alias (
  trope_id  TEXT NOT NULL,
  alias     TEXT NOT NULL,
  priority  INTEGER DEFAULT 100,
  is_blocked INTEGER DEFAULT 0, -- 0/1
  PRIMARY KEY (trope_id, alias),
  FOREIGN KEY (trope_id) REFERENCES trope(id) ON DELETE CASCADE
);

-- ===== FTS triggers =====
CREATE TRIGGER IF NOT EXISTS chunk_fts_after_insert
AFTER INSERT ON chunk
BEGIN
  INSERT INTO chunk_fts(rowid, text) VALUES (new.rowid, new.text);
END;

CREATE TRIGGER IF NOT EXISTS chunk_fts_after_delete
AFTER DELETE ON chunk
BEGIN
  INSERT INTO chunk_fts(chunk_fts, rowid, text) VALUES ('delete', old.rowid, old.text);
END;

CREATE TRIGGER IF NOT EXISTS chunk_fts_after_update_text
AFTER UPDATE OF text ON chunk
BEGIN
  INSERT INTO chunk_fts(chunk_fts, rowid, text) VALUES ('delete', old.rowid, old.text);
  INSERT INTO chunk_fts(rowid, text) VALUES (new.rowid, new.text);
END;

-- ===== Helpful view =====
CREATE VIEW IF NOT EXISTS v_recent_findings AS
SELECT f.*, t.name AS trope_name
FROM trope_finding f
JOIN trope t ON t.id = f.trope_id
ORDER BY f.created_at DESC;

-- ===== One-time cleanup before enforcing uniqueness =====
-- Safe on empty/new DBs; on legacy DBs removes dup rows.
DELETE FROM trope_candidate
WHERE rowid NOT IN (
  SELECT MIN(rowid)
  FROM trope_candidate
  GROUP BY work_id, trope_id, start, end
);

DELETE FROM trope_finding
WHERE rowid NOT IN (
  SELECT MIN(rowid)
  FROM trope_finding
  GROUP BY work_id, trope_id, evidence_start, evidence_end
);

-- ===== Enforce uniqueness (define once, after cleanup) =====
-- Candidate: unique per (work, trope, exact char span)
CREATE UNIQUE INDEX IF NOT EXISTS uniq_candidate_span
  ON trope_candidate(work_id, trope_id, start, end);

-- Finding: unique per (work, trope, evidence span)
CREATE UNIQUE INDEX IF NOT EXISTS uq_finding_span
  ON trope_finding(work_id, trope_id, evidence_start, evidence_end);
