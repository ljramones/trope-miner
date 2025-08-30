#!/usr/bin/env python3
"""
Load/Upsert tropes from CSV into SQLite.

CSV expected headers:
  required: name
  optional: id, summary|description|definition, source_url|source|url|link, aliases

Aliases:
  - If the CSV 'aliases' cell starts with '[', it's treated as JSON (array of strings).
  - Otherwise it's split on | ; , and normalized.
  - Cleaned aliases are stored as JSON **only if non-empty**; otherwise aliases is set to NULL.

Examples:
  python load_tropes.py --db ./tropes.db --csv ./tropes.csv
  python load_tropes.py --db ./tropes.db --csv ./tropes.csv --clear
"""
import argparse
import csv
import json
import re
import sqlite3
import unicodedata
from pathlib import Path
from typing import List, Optional


def slugify(text: str) -> str:
    t = unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode()
    t = re.sub(r"[^A-Za-z0-9\s-]", " ", t)
    t = re.sub(r"\s+", "-", t).strip("-").lower()
    t = re.sub(r"-+", "-", t)
    return t or "trope"


def ascii_fold(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")


def normalize_alias(s: str) -> str:
    s = ascii_fold(s.strip().lower())
    s = re.sub(r"\s+", " ", s)
    s = s.strip(",.;:!?\"'()[]{}")
    # keep short but useful; avoid ultra-long phrases
    if len(s) > 40:
        s = s[:40].rstrip()
    return s


def parse_aliases_field(raw: Optional[str]) -> List[str]:
    """Return a cleaned, deduped list of alias strings."""
    if not raw:
        return []
    raw = raw.strip()
    # JSON array?
    if raw.startswith("["):
        try:
            data = json.loads(raw)
            if isinstance(data, list):
                cand = [x for x in data if isinstance(x, str)]
            else:
                cand = []
        except Exception:
            cand = []
    else:
        # Legacy split on | ; , (tolerate mixed use)
        parts = re.split(r"[|;,]", raw)
        cand = [p for p in (x.strip() for x in parts) if p]

    out, seen = [], set()
    for a in cand:
        n = normalize_alias(a)
        if not n or n in seen:
            continue
        # filter obvious junk
        if n in {"trope", "cliche", "stereotype"}:
            continue
        out.append(n)
        seen.add(n)
        if len(out) >= 24:
            break
    return out


def pick(row: dict, *keys: str) -> Optional[str]:
    """Pick first non-empty value from possible column names."""
    for k in keys:
        if k in row:
            v = (row.get(k) or "").strip()
            if v:
                return v
    return None


def ensure_trope_table(conn: sqlite3.Connection) -> None:
    # Minimal guard; your main schema is created elsewhere.
    conn.execute("""
        CREATE TABLE IF NOT EXISTS trope(
          id         TEXT PRIMARY KEY,
          name       TEXT NOT NULL UNIQUE,
          summary    TEXT,
          source_url TEXT,
          aliases    TEXT
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_trope_name ON trope(name)")
    conn.commit()


def main():
    ap = argparse.ArgumentParser(description="Load/Upsert tropes from CSV")
    ap.add_argument("--db", required=True)
    ap.add_argument("--csv", required=True, help="CSV with columns: id,name,summary,source_url,aliases")
    ap.add_argument("--clear", action="store_true", help="DELETE FROM trope before loading")
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    conn.execute("PRAGMA foreign_keys=ON")
    ensure_trope_table(conn)

    if args.clear:
        conn.execute("DELETE FROM trope")
        conn.commit()

    rows = 0
    inserted = 0
    updated = 0

    # UTF-8 with possible BOM
    with Path(args.csv).open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if "name" not in (reader.fieldnames or []):
            raise SystemExit("CSV must include a 'name' column.")

        for r in reader:
            name = (r.get("name") or "").strip()
            if not name:
                continue

            tid = (r.get("id") or "").strip() or slugify(name)
            summary = pick(r, "summary", "description", "definition", "text", "body")
            src = pick(r, "source_url", "source", "url", "link")
            aliases_raw = pick(r, "aliases")

            aliases_list = parse_aliases_field(aliases_raw)
            aliases_json = json.dumps(aliases_list, ensure_ascii=True) if aliases_list else None

            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO trope(id, name, summary, source_url, aliases)
                VALUES(?,?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET
                  name=excluded.name,
                  summary=excluded.summary,
                  source_url=excluded.source_url,
                  aliases=excluded.aliases
                """,
                (tid, name, summary, src, aliases_json),
            )
            # crude insert vs update heuristic: if name didn't exist before and rowid changed
            # SQLite doesn't report affected rows distinctly; we'll just count all as upserts
            rows += 1
            # Optionally check if it existed by querying first, but that’s slower.

    conn.commit()
    print(f"[load_tropes] upserted {rows} rows into trope")

    conn.close()


if __name__ == "__main__":
    main()
