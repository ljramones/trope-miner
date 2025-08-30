#!/usr/bin/env python3
"""
Trope Miner — Ingest & Segmenter
--------------------------------

Purpose
  • Ingest raw fiction text files (UTF-8) into SQLite as the system of record.
  • Normalize text, detect chapters & scenes, and build 300–600 "token" chunks with 60–100 overlap.
  • Persist offsets for explainability; prepare data for later embedding in Chroma.

Usage (examples)
  $ python ingester_segmenter.py ingest \
      --db ./tropes.db \
      --file ./samples/novella.txt \
      --title "Novella Title" --author "A. Writer" \
      --target 450 --overlap 80

  # Preview only (no writes):
  $ python ingester_segmenter.py ingest --db ./tropes.db --file ./samples/novella.txt --dry-run

  # Show brief stats after ingest:
  $ python ingester_segmenter.py stats --db ./tropes.db --work-id <uuid>

Design notes
  • Uses SQLite WAL mode for concurrency and speed.
  • Stores raw and normalized text; offsets are computed against normalized text.
  • Minimal deps (stdlib + optional 'regex' if you prefer). No external tokenizers.
  • "Tokens" here ≈ whitespace-delimited words; good enough for 300–600 targets.
  • You can later swap in a tiktoken/HF tokenizer by implementing Tokenizer.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import sqlite3
import sys
import textwrap
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

DB_DDL = """
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
  sha256      TEXT,
  FOREIGN KEY(work_id)  REFERENCES work(id)  ON DELETE CASCADE,
  FOREIGN KEY(scene_id) REFERENCES scene(id) ON DELETE SET NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_chunk_sha256      ON chunk(sha256);
CREATE INDEX        IF NOT EXISTS idx_chunk_work_idx    ON chunk(work_id, idx);
CREATE INDEX        IF NOT EXISTS idx_chunk_work_scene  ON chunk(work_id, scene_id, idx);
CREATE INDEX        IF NOT EXISTS idx_chunk_scene       ON chunk(scene_id);
CREATE INDEX        IF NOT EXISTS idx_chunk_work_span   ON chunk(work_id, char_start, char_end);

-- Embedding backref (vectors live in Chroma)
CREATE TABLE IF NOT EXISTS embedding_ref (
  chunk_id   TEXT    NOT NULL,
  collection TEXT    NOT NULL,
  model      TEXT    NOT NULL,
  dim        INTEGER NOT NULL,
  chroma_id  TEXT    NOT NULL,
  PRIMARY KEY (chunk_id, collection),
  FOREIGN KEY (chunk_id) REFERENCES chunk(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_embedding_model ON embedding_ref(model);

-- Optional: FTS mirror of chunk.text
CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(
  text,
  content='chunk',
  content_rowid='rowid'
);

-- Keep FTS in sync when chunk.text changes or is deleted
CREATE TRIGGER IF NOT EXISTS chunk_fts_after_delete
AFTER DELETE ON chunk BEGIN
  INSERT INTO chunk_fts(chunk_fts, rowid, text)
  VALUES('delete', old.rowid, old.text);
END;

CREATE TRIGGER IF NOT EXISTS chunk_fts_after_update_text
AFTER UPDATE OF text ON chunk BEGIN
  INSERT INTO chunk_fts(chunk_fts, rowid, text)
  VALUES('delete', old.rowid, old.text);
  INSERT INTO chunk_fts(rowid, text)
  VALUES(new.rowid, new.text);
END;
"""

# ----------------------------- Tokenization -----------------------------

@dataclass
class TokenSpan:
    start: int  # char offset (inclusive)
    end: int    # char offset (exclusive)
    text: str

class Tokenizer:
    """Whitespace & punctuation aware tokenizer with char offsets."""
    _tok_re = re.compile(r"\w+|[^\w\s]", re.UNICODE)

    def tokenize(self, text: str) -> List[TokenSpan]:
        return [TokenSpan(m.start(), m.end(), m.group(0)) for m in self._tok_re.finditer(text)]

# ----------------------------- Normalization ----------------------------

def normalize_text(raw: str) -> str:
    # Normalize newlines, collapse trailing spaces; keep paragraph breaks
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    # Collapse 3+ blank lines to max 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text

# ----------------------------- Chapter & Scene detection ----------------

_CHAPTER_LINE_RE = re.compile(r"^(?:chapter|ch\.|book|part)\s+([ivxlcdm]+|\d+|[a-z]+)\b",
                              re.IGNORECASE)
_SCENE_RULE = re.compile(r"^(?:\*{3,}|-{3,}|—{2,}|·{3,}|\*\s\*\s\*|#\s#\s#)\s*$")

@dataclass
class Span:
    start: int
    end: int
    title: Optional[str] = None

@dataclass
class Scene:
    start: int
    end: int
    heading: Optional[str]

def detect_chapters(text: str) -> List[Span]:
    """Heuristic: lines beginning with Chapter/Book/Part ... else single chapter covering all."""
    lines = text.split("\n")
    offsets = []
    pos = 0
    for line in lines:
        offsets.append(pos)
        pos += len(line) + 1  # +1 for newline

    candidates: List[Tuple[int, str]] = []
    for i, line in enumerate(lines):
        if _CHAPTER_LINE_RE.search(line.strip()):
            candidates.append((i, line.strip()))

    if not candidates:
        return [Span(0, len(text), title=None)]

    spans: List[Span] = []
    for idx, (line_idx, title) in enumerate(candidates):
        start = offsets[line_idx]
        end = offsets[candidates[idx + 1][0]] if idx + 1 < len(candidates) else len(text)
        spans.append(Span(start, end, title=title))
    return spans

def detect_scenes(chapter_text: str, chapter_start: int) -> List[Scene]:
    """Scenes separated by blank-line markers or common ornament separators."""
    lines = chapter_text.split("\n")
    # Build line start offsets relative to full doc
    line_offsets: List[int] = []
    pos = chapter_start
    for line in lines:
        line_offsets.append(pos)
        pos += len(line) + 1

    # Identify separator lines
    sep_idxs = []
    for i, line in enumerate(lines):
        if _SCENE_RULE.match(line.strip()):
            sep_idxs.append(i)
        # Also consider 2+ consecutive blank lines as soft scene break
        if i + 1 < len(lines) and not lines[i].strip() and not lines[i + 1].strip():
            sep_idxs.append(i + 1)

    # always include start and end as anchors
    anchors = sorted(set([0] + sep_idxs + [len(lines)]))
    scenes: List[Scene] = []
    for a, b in zip(anchors[:-1], anchors[1:]):
        start = line_offsets[a]
        end = line_offsets[b] if b < len(line_offsets) else chapter_start + len(chapter_text)
        # Skip empty spans
        if end > start and chapter_text[start - chapter_start:end - chapter_start].strip():
            heading = lines[a].strip() if _SCENE_RULE.match(lines[a].strip()) else None
            scenes.append(Scene(start, end, heading=heading))

    if not scenes:
        scenes = [Scene(chapter_start, chapter_start + len(chapter_text), heading=None)]
    return scenes

# ----------------------------- Chunking ---------------------------------

@dataclass
class Chunk:
    start: int
    end: int
    token_start: int
    token_end: int
    text: str

def chunk_scene(text: str, scene: Scene, tokenizer: Tokenizer, target: int = 450,
                min_tokens: int = 300, max_tokens: int = 600, overlap: int = 80) -> List[Chunk]:
    """Create overlapping chunks within [scene.start, scene.end]."""
    doc_span = text[scene.start:scene.end]
    tokens = tokenizer.tokenize(doc_span)
    if not tokens:
        return []

    chunks: List[Chunk] = []
    i = 0
    N = len(tokens)
    while i < N:
        # window [i, j)
        j = min(i + target, N)
        # Expand to min_tokens if target is too small near the beginning
        j = max(j, min(N, i + min_tokens))
        # Clamp to max_tokens
        j = min(i + max_tokens, j, N)

        t_start = tokens[i].start
        t_end = tokens[j - 1].end
        start = scene.start + t_start
        end = scene.start + t_end

        # Attempt to extend to nearest sentence boundary (simple heuristic)
        extend = j
        k = min(N - 1, j + 30)
        boundary_found = False
        for p in range(j, k + 1):  # inclusive window
            if tokens[p].text in {'.', '!', '?'}:
                extend = p + 1
                boundary_found = True
                break
        if boundary_found:
            extend = min(extend, i + max_tokens)  # don't exceed max_tokens
            t_end = tokens[extend - 1].end
            end = scene.start + t_end
            j = extend

        text_chunk = text[start:end]
        chunks.append(Chunk(start=start, end=end, token_start=i, token_end=j, text=text_chunk))

        if j >= N:
            break
        # Overlap
        i = max(0, j - overlap)

    return chunks

# ----------------------------- File decoding (robust) -------------------

def read_text_smart(file_path: Path, encoding: Optional[str] = "auto") -> str:
    """Read text with BOM/heuristic detection.
    Supports: utf-8/utf-8-sig, utf-16(le/be), utf-32(le/be), cp1252, latin-1.
    If encoding != 'auto', decode with the given codec.
    """
    data = file_path.read_bytes()
    if encoding and encoding != "auto":
        return data.decode(encoding)

    # BOM-based detection (order matters!)
    # UTF-8 BOM
    if data.startswith(b"\xef\xbb\xbf"):
        return data.decode("utf-8-sig")
    # UTF-32 BOMs
    if data.startswith(b"\xff\xfe\x00\x00"):
        return data.decode("utf-32-le")
    if data.startswith(b"\x00\x00\xfe\xff"):
        return data.decode("utf-32-be")
    # UTF-16 BOMs
    if data.startswith(b"\xff\xfe"):
        return data.decode("utf-16-le")
    if data.startswith(b"\xfe\xff"):
        return data.decode("utf-16-be")

    # Heuristic: lots of NULs suggests UTF-16/32 without BOM
    sample = data[:4096]
    nul_ratio = sample.count(b"\x00") / max(1, len(sample))
    if nul_ratio > 0.30:
        for enc in ("utf-16-le", "utf-16-be", "utf-32-le", "utf-32-be"):
            try:
                return data.decode(enc)
            except UnicodeDecodeError:
                continue

    # Try UTF-8, then cp1252, then mac_roman, then latin-1 as last resort
    for enc in ("utf-8", "cp1252", "mac_roman", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue

    # Fallback with replacement to avoid hard failure
    return data.decode("utf-8", errors="replace")

# ----------------------------- Persistence ------------------------------

def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(DB_DDL)
    conn.commit()

def insert_work(conn: sqlite3.Connection, title: str, author: str,
                raw_text: str, norm_text: str, char_count: int,
                source: str = "local", license: str = "unknown",
                work_id: Optional[str] = None) -> str:
    wid = work_id or str(uuid.uuid4())
    conn.execute(
        "INSERT INTO work(id, title, author, source, license, raw_text, norm_text, char_count) "
        "VALUES(?,?,?,?,?,?,?,?)",
        (wid, title, author, source, license, raw_text, norm_text, char_count)
    )
    return wid

def insert_chapters(conn: sqlite3.Connection, work_id: str, chapters: List[Span]) -> List[str]:
    ids: List[str] = []
    for idx, ch in enumerate(chapters):
        cid = str(uuid.uuid4())
        conn.execute(
            "INSERT INTO chapter(id, work_id, idx, title, char_start, char_end) VALUES(?,?,?,?,?,?)",
            (cid, work_id, idx, ch.title, ch.start, ch.end)
        )
        ids.append(cid)
    return ids

def insert_scenes(conn: sqlite3.Connection, work_id: str, chapter_id: str, scenes: List[Scene]) -> List[str]:
    ids: List[str] = []
    for idx, sc in enumerate(scenes):
        sid = str(uuid.uuid4())
        conn.execute(
            "INSERT INTO scene(id, work_id, chapter_id, idx, char_start, char_end, heading) VALUES(?,?,?,?,?,?,?)",
            (sid, work_id, chapter_id, idx, sc.start, sc.end, sc.heading)
        )
        ids.append(sid)
    return ids

def insert_chunks(conn: sqlite3.Connection, work_id: str, scene_id: str, chunks: List[Chunk]) -> List[str]:
    ids: List[str] = []
    rows = []
    for idx, c in enumerate(chunks):
        cid = str(uuid.uuid4())
        sha = hashlib.sha256(c.text.encode("utf-8")).hexdigest()
        rows.append((cid, work_id, scene_id, idx, c.start, c.end, c.token_start, c.token_end, c.text, sha))
        ids.append(cid)
    conn.executemany(
        "INSERT INTO chunk(id, work_id, scene_id, idx, char_start, char_end, token_start, token_end, text, sha256) "
        "VALUES(?,?,?,?,?,?,?,?,?,?)",
        rows
    )
    # populate FTS
    conn.executemany(
        "INSERT INTO chunk_fts(rowid, text) SELECT rowid, text FROM chunk WHERE id = ?",
        [(cid,) for cid in ids]
    )
    return ids

# ----------------------------- Ingest flow ------------------------------

@dataclass
class IngestStats:
    chapters: int
    scenes: int
    chunks: int

def ingest_text(conn: sqlite3.Connection, file_path: Path, title: str, author: str,
                source: str = "local", license: str = "unknown",
                target: int = 450, overlap: int = 80,
                min_tokens: int = 300, max_tokens: int = 600,
                dry_run: bool = False, encoding: Optional[str] = "auto") -> Tuple[str, IngestStats]:
    raw = read_text_smart(file_path, encoding=encoding)
    norm = normalize_text(raw)
    char_count = len(norm)

    chapters = detect_chapters(norm)
    tokenizer = Tokenizer()

    if dry_run:
        # Quick preview
        total_scenes = 0
        total_chunks = 0
        for ch in chapters:
            scs = detect_scenes(norm[ch.start:ch.end], ch.start)
            total_scenes += len(scs)
            for sc in scs:
                chunks = chunk_scene(norm, sc, tokenizer, target=target, min_tokens=min_tokens,
                                     max_tokens=max_tokens, overlap=overlap)
                total_chunks += len(chunks)
        return "DRY-RUN", IngestStats(len(chapters), total_scenes, total_chunks)

    work_id = insert_work(conn,
                          title=title,
                          author=author,
                          raw_text=raw,
                          norm_text=norm,
                          char_count=char_count,
                          source=source,
                          license=license)

    # Back-compat: if an older DB lacks char_count, ignore failure.
    try:
        conn.execute("UPDATE work SET char_count=? WHERE id=?", (char_count, work_id))
    except sqlite3.OperationalError:
        pass

    chapter_ids = insert_chapters(conn, work_id, chapters)

    total_scenes = 0
    total_chunks = 0

    for ch, cid in zip(chapters, chapter_ids):
        scs = detect_scenes(norm[ch.start:ch.end], ch.start)
        scene_ids = insert_scenes(conn, work_id, cid, scs)
        total_scenes += len(scs)
        for sc, sid in zip(scs, scene_ids):
            chunks = chunk_scene(norm, sc, tokenizer, target=target, min_tokens=min_tokens,
                                 max_tokens=max_tokens, overlap=overlap)
            insert_chunks(conn, work_id, sid, chunks)
            total_chunks += len(chunks)

    conn.commit()
    return work_id, IngestStats(len(chapters), total_scenes, total_chunks)

# ----------------------------- CLI --------------------------------------

def cmd_ingest(args: argparse.Namespace) -> None:
    db = sqlite3.connect(args.db)
    ensure_schema(db)
    work_id, stats = ingest_text(
        db,
        file_path=Path(args.file),
        title=args.title or Path(args.file).stem,
        author=args.author or "",
        source=args.source,
        license=args.license,
        target=args.target,
        overlap=args.overlap,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
        dry_run=args.dry_run,
        encoding=args.encoding,
    )
    if args.dry_run:
        print(f"[DRY-RUN] chapters={stats.chapters} scenes={stats.scenes} chunks={stats.chunks}")
    else:
        print(f"[OK] work_id={work_id} chapters={stats.chapters} scenes={stats.scenes} chunks={stats.chunks}")
    db.close()

def cmd_stats(args: argparse.Namespace) -> None:
    con = sqlite3.connect(args.db)
    cur = con.cursor()
    if args.work_id:
        cur.execute("SELECT title, author FROM work WHERE id=?", (args.work_id,))
        row = cur.fetchone()
        if not row:
            print("work not found", file=sys.stderr)
            con.close()
            return
        title, author = row
        cur.execute("SELECT COUNT(*) FROM chapter WHERE work_id=?", (args.work_id,))
        chapters = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM scene WHERE work_id=?", (args.work_id,))
        scenes = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM chunk WHERE work_id=?", (args.work_id,))
        chunks = cur.fetchone()[0]
        print(f"work={title!r} author={author!r} chapters={chapters} scenes={scenes} chunks={chunks}")
    else:
        cur.execute("SELECT COUNT(*) FROM work")
        w = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM chapter")
        c = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM scene")
        s = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM chunk")
        k = cur.fetchone()[0]
        print(f"works={w} chapters={c} scenes={s} chunks={k}")
    con.close()

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ingester_segmenter",
                                description="Ingest raw text and segment to chapters/scenes/chunks")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_ing = sub.add_parser("ingest", help="Ingest a text file into SQLite and segment it")
    p_ing.add_argument("--db", required=True)
    p_ing.add_argument("--file", required=True)
    p_ing.add_argument("--title")
    p_ing.add_argument("--author")
    p_ing.add_argument("--source", default="local")
    p_ing.add_argument("--license", default="unknown")
    p_ing.add_argument("--target", type=int, default=450)
    p_ing.add_argument("--overlap", type=int, default=80)
    p_ing.add_argument("--min-tokens", type=int, default=300)
    p_ing.add_argument("--max-tokens", type=int, default=600)
    p_ing.add_argument("--encoding", default="auto", help="Text encoding (e.g., utf-8, utf-16-le, cp1252) or 'auto'")
    p_ing.add_argument("--dry-run", action="store_true")
    p_ing.set_defaults(func=cmd_ingest)

    p_stats = sub.add_parser("stats", help="Show DB stats or a specific work's stats")
    p_stats.add_argument("--db", required=True)
    p_stats.add_argument("--work-id")
    p_stats.set_defaults(func=cmd_stats)

    return p

def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    args.func(args)

if __name__ == "__main__":
    main()

