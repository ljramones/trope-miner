#!/usr/bin/env python3
r"""
Seed trope candidates using word-boundary regex matches on chunk text.

- Reads aliases from trope.aliases (JSON array) and ALSO uses the canonical trope name.
- Skips ultra-generic/short aliases (stoplist + min length), BUT never drops the canonical name.
- Uses chunk-local regex matches; converts to work-level offsets with chunk.char_start.
- Inserts into trope_candidate(
      id, work_id, scene_id, chunk_id, trope_id, surface, alias,
      start, end, source='gazetteer', score=0.0
  )
- Idempotent across runs via a UNIQUE index on (work_id, trope_id, start, end).

Usage:
  python scripts/seed_candidates_boundary.py --db ./tropes.db --work-id <UUID> \
      [--min-len 5] [--max-per-trope 500] [--stoplist extra_stopwords.txt]
"""

import argparse
import json
import re
import sqlite3
import sys
from typing import Dict, List, Set, Tuple

# ----------------------------------------------------------------------
# Stoplist of overly-generic single words / short phrases that caused noise.
# NOTE: Canonical trope names are ALWAYS kept even if they appear here.
#       This stoplist only applies to non-canonical aliases.
# ----------------------------------------------------------------------
STOPLIST: Set[str] = {
    # very generic one-worders
    "hero", "villain", "power", "fight", "battle", "magic", "love", "war",
    "secret", "plan", "agent", "mystery", "weapon", "girl", "boy",
    "night", "day", "city", "king", "queen", "man", "woman", "monster", "beast",
    "darkness", "light", "death", "life", "friend", "enemy", "revenge", "curse",

    # previously noisy LLM-suggested aliases we want to block globally
    "buddy", "backup", "job", "serious", "calm", "opposite", "haunted",
    "first glance",
    # (intentionally NOT including "detective" here)
}

def load_stoplist(path: str) -> Set[str]:
    s: Set[str] = set()
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                w = line.strip().lower()
                if not w or w.startswith("#"):
                    continue
                s.add(w)
    except Exception as e:
        print(f"[warn] could not read stoplist {path}: {e}", file=sys.stderr)
    return s

# ----------------------------------------------------------------------
# DB access
# ----------------------------------------------------------------------
def load_tropes(conn: sqlite3.Connection) -> List[Dict]:
    cur = conn.cursor()
    rows = cur.execute(
        "SELECT id, name, aliases FROM trope ORDER BY name COLLATE NOCASE"
    ).fetchall()
    out: List[Dict] = []
    for tid, name, aliases_json in rows:
        aliases: List[str] = []
        if aliases_json:
            try:
                arr = json.loads(aliases_json)
                if isinstance(arr, list):
                    aliases = [a for a in arr if isinstance(a, str)]
            except Exception:
                # ignore bad JSON
                pass
        out.append({"id": tid, "name": name, "aliases": aliases})
    return out


def load_chunks(conn: sqlite3.Connection, work_id: str):
    cur = conn.cursor()
    rows = cur.execute(
        "SELECT id, scene_id, char_start, char_end, text "
        "FROM chunk WHERE work_id=? ORDER BY char_start ASC",
        (work_id,),
    ).fetchall()
    return rows


# ----------------------------------------------------------------------
# Normalization & filtering
# ----------------------------------------------------------------------
def norm_alias(a: str) -> str:
    a = a.strip().lower()
    a = re.sub(r"\s+", " ", a)  # collapse whitespace
    a = a.strip(",.;:!?\"'()[]{}")  # trim common punctuation at ends
    return a


def alias_ok(alias: str, min_len: int) -> bool:
    if not alias:
        return False
    if len(alias) < min_len:
        return False
    if alias in STOPLIST:
        return False
    return True


# ----------------------------------------------------------------------
# Regex builder
#   - Word-boundary style lookarounds at the edges.
#   - Flexible whitespace within phrases (\s+).
#   - Treat ASCII hyphen and unicode dashes as equivalent.
#   - Normalize ASCII vs curly apostrophes.
# ----------------------------------------------------------------------
_DASH_CLASS = r"[-\u2010-\u2015]"  # hyphen + Unicode hyphen/dash range


def _escape_token(token: str) -> str:
    """Escape a token and normalize punctuation variants (dashes, apostrophes)."""
    esc = re.escape(token)
    # Normalize hyphen/dash variants *inside* the token so alias “Face–Heel” matches Face-Heel/Face—Heel
    esc = (
        esc.replace("-", _DASH_CLASS)
           .replace("–", _DASH_CLASS)
           .replace("—", _DASH_CLASS)
           .replace(r"\-", _DASH_CLASS)
    )
    # Normalize ASCII vs curly apostrophes inside the token (e.g., Chekhov's vs Chekhov’s)
    esc = esc.replace("'", r"['\u2019]").replace("’", r"['\u2019]")
    return esc


def build_pattern(alias: str) -> re.Pattern:
    r"""
    Build a case-insensitive pattern for an alias:
      - edges: word-boundary-like lookarounds
      - internal whitespace → \s+; internal hyphens/dashes → [-\u2010-\u2015\s]+
      - single-word alphabetic aliases get an optional plural (s|es)
      - multi-word aliases also allow an optional plural on the final alphabetic word
    """
    a = alias.strip()
    parts = re.split(r"\s+", a)
    esc = [_escape_token(p) for p in parts if p]
    if not esc:
        return re.compile(r"(?!)")  # never matches

    if len(esc) == 1 and re.fullmatch(r"[A-Za-z]+", parts[0]):
        # optional plural for simple words
        core = rf"{esc[0]}(?:s|es)?"
    else:
        # allow spaces OR any Unicode dash between parts
        joiner = rf"{_DASH_CLASS}+\s*|\s+"
        # join with either dash+optional space OR plain whitespace (use a non-capturing group)
        joiner_group = rf"(?:{joiner})"
        core = joiner_group.join(esc)

        # Optional plural on the last alphabetic word in multiword aliases (e.g., "bottle episode(s)")
        last_src = parts[-1]
        if re.fullmatch(r"[A-Za-z]+", last_src):
            esc_last_plural = rf"(?:{esc[-1]}(?:s|es)?)"
            core = joiner_group.join([*esc[:-1], esc_last_plural])

    return re.compile(rf"(?<!\w){core}(?!\w)", re.IGNORECASE)


# ----------------------------------------------------------------------
# Indexes & uniqueness
# ----------------------------------------------------------------------
def ensure_indexes(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    # helpful lookups
    cur.execute("CREATE INDEX IF NOT EXISTS idx_candidate_work  ON trope_candidate(work_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_candidate_trope ON trope_candidate(trope_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_chunk_work_span ON chunk(work_id, char_start, char_end)")
    # protect against duplicates across runs
    cur.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_candidate_span "
        "ON trope_candidate(work_id, trope_id, start, end)"
    )
    conn.commit()


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Seed candidates with word-boundary alias matching.")
    ap.add_argument("--db", required=True, help="Path to tropes.db")
    ap.add_argument("--work-id", required=True, help="Work UUID to scan")
    ap.add_argument("--min-len", type=int, default=5,
                    help="Minimum alias length for non-canonical aliases (default: 5)")
    ap.add_argument("--stoplist", help="Path to newline-delimited extra stopwords")
    ap.add_argument("--max-per-trope", type=int, default=500, help="Cap inserts per trope (safety)")
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    ensure_indexes(conn)

    tropes = load_tropes(conn)
    chunks = load_chunks(conn, args.work_id)

    if args.stoplist:
        # mutate in place to avoid needing 'global'
        STOPLIST.update(load_stoplist(args.stoplist))

    if not chunks:
        print("No chunks found for work:", args.work_id, file=sys.stderr)
        sys.exit(1)

    cur = conn.cursor()
    total_inserts = 0

    for trope in tropes:
        tid = trope["id"]

        # Always include canonical name, even if "generic"
        canon = norm_alias(trope["name"])
        entries: List[Tuple[str, bool]] = [(canon, True)]

        # Add normalized non-canonical aliases
        entries.extend((norm_alias(a), False) for a in trope["aliases"])

        # Filter: keep canonical; apply stoplist/min_len to others
        alias_list: List[str] = []
        for alias, is_canon in entries:
            if not alias:
                continue
            if is_canon:
                alias_list.append(alias)
            else:
                if alias_ok(alias, args.min_len):
                    alias_list.append(alias)

        # Stable de-dup
        alias_list = list(dict.fromkeys(alias_list))
        if not alias_list:
            continue

        compiled = [(a, build_pattern(a)) for a in alias_list]

        per_trope = 0
        seen_spans: Set[Tuple[str, int, int]] = set()  # (trope_id, start, end)

        for chunk_row in chunks:
            chunk_id = chunk_row["id"]
            scene_id = chunk_row["scene_id"]
            ch_start = chunk_row["char_start"]
            ch_end = chunk_row["char_end"]
            text = chunk_row["text"] or ""
            if not text:
                continue

            for alias, pat in compiled:
                for m in pat.finditer(text):
                    start = ch_start + m.start()
                    end = ch_start + m.end()
                    key = (tid, start, end)
                    if key in seen_spans:
                        continue

                    # sanity: ensure span fits inside the chunk’s range
                    if start < ch_start or end > ch_end:
                        continue

                    try:
                        cur.execute(
                            "INSERT OR IGNORE INTO trope_candidate (id, work_id, scene_id, chunk_id, trope_id, surface, alias, start, end, source, score) "
                            "VALUES (lower(hex(randomblob(16))), ?, ?, ?, ?, ?, ?, ?, ?, 'gazetteer', 0.0)",
                            (args.work_id, scene_id, chunk_id, tid, m.group(0), alias, start, end)
                        )
                        # If actually inserted (not ignored by UNIQUE)
                        if cur.rowcount:
                            total_inserts += 1
                            per_trope += 1
                            seen_spans.add(key)
                    except sqlite3.IntegrityError:
                        # Should be rare due to OR IGNORE + UNIQUE index
                        pass

                    if per_trope >= args.max_per_trope:
                        break
                if per_trope >= args.max_per_trope:
                    break
            if per_trope >= args.max_per_trope:
                break

    conn.commit()
    print(f"Seeded {total_inserts} boundary-matched candidate hits for work {args.work_id}")
    conn.close()


if __name__ == "__main__":
    main()
