#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json, sqlite3, sys, uuid
from pathlib import Path
from typing import List, Optional

# ------------------------------- schema --------------------------------

def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA foreign_keys=ON;")

    # Create trope table if missing; otherwise add optional columns
    has_trope = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='trope' LIMIT 1"
    ).fetchone() is not None

    if not has_trope:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS trope (
          id            TEXT PRIMARY KEY,
          name          TEXT NOT NULL UNIQUE,
          summary       TEXT,
          long_desc     TEXT,
          tags          TEXT,   -- JSON array
          source_url    TEXT,
          aliases       TEXT,   -- JSON array
          anti_aliases  TEXT,   -- JSON array
          tvtropes_url  TEXT,
          updated_at    TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
        );
        CREATE INDEX IF NOT EXISTS idx_trope_name ON trope(name);
        """)
    else:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(trope)")}
        def add(col: str, decl: str):
            if col not in cols:
                conn.execute(f"ALTER TABLE trope ADD COLUMN {col} {decl}")
        # These might be missing on older DBs; add them if needed
        add("summary", "TEXT")
        add("long_desc", "TEXT")
        add("tags", "TEXT")
        add("source_url", "TEXT")
        add("aliases", "TEXT")
        add("anti_aliases", "TEXT")
        add("tvtropes_url", "TEXT")
        add("updated_at", "TEXT")  # default handled in INSERT/UPDATE below
        conn.execute("CREATE INDEX IF NOT EXISTS idx_trope_name ON trope(name)")

    # Group tables for ontology (safe to create if they already exist)
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS trope_group(
      id   TEXT PRIMARY KEY,
      name TEXT NOT NULL
    );
    CREATE UNIQUE INDEX IF NOT EXISTS uq_trope_group_name ON trope_group(name);

    CREATE TABLE IF NOT EXISTS trope_group_member(
      trope_id TEXT NOT NULL,
      group_id TEXT NOT NULL,
      PRIMARY KEY (trope_id, group_id)
    );
    """)
    conn.commit()

# ------------------------------ helpers -------------------------------

def jdump_list(v: Optional[List[str]]) -> str:
    return json.dumps([x for x in (v or []) if isinstance(x, str) and x.strip()])

def split_field(s: Optional[str]) -> List[str]:
    """Accept JSON array or pipe/semicolon-delimited strings."""
    if not s:
        return []
    s = s.strip()
    if s.startswith("[") and s.endswith("]"):
        try:
            arr = json.loads(s)
            return [x for x in arr if isinstance(x, str)]
        except Exception:
            pass
    parts = []
    for tok in [p.strip() for p in s.replace(";", "|").split("|")]:
        if tok:
            parts.append(tok)
    return parts

def get_or_create_group(conn: sqlite3.Connection, name: str) -> str:
    r = conn.execute(
        "SELECT id FROM trope_group WHERE name=? COLLATE NOCASE", (name,)
    ).fetchone()
    if r:
        return r[0]
    gid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"group:{name.lower()}"))
    conn.execute("INSERT OR IGNORE INTO trope_group(id,name) VALUES (?,?)", (gid, name))
    return gid

def upsert_trope(conn: sqlite3.Connection, row: dict) -> None:
    tid   = (row.get("id") or "").strip()
    name  = (row.get("name") or "").strip()
    if not name:
        return
    if not tid:
        tid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"trope:{name.lower()}"))

    summary   = (row.get("summary") or "").strip()
    long_desc = (row.get("long_desc") or "").strip()
    source    = (row.get("source_url") or "").strip()
    tvt       = (row.get("tvtropes_url") or "").strip()

    aliases = split_field(row.get("aliases"))
    anti    = split_field(row.get("anti_aliases"))
    tags    = split_field(row.get("tags"))

    conn.execute(
        """
        INSERT INTO trope(id,name,summary,long_desc,tags,source_url,aliases,anti_aliases,tvtropes_url,updated_at)
        VALUES(?,?,?,?,?,?,?,?,?,strftime('%Y-%m-%dT%H:%M:%fZ','now'))
        ON CONFLICT(id) DO UPDATE SET
          name=excluded.name,
          summary=excluded.summary,
          long_desc=excluded.long_desc,
          tags=excluded.tags,
          source_url=excluded.source_url,
          aliases=excluded.aliases,
          anti_aliases=excluded.anti_aliases,
          tvtropes_url=excluded.tvtropes_url,
          updated_at=strftime('%Y-%m-%dT%H:%M:%fZ','now')
        """,
        (tid, name, summary, long_desc, jdump_list(tags), source,
         jdump_list(aliases), jdump_list(anti), tvt),
    )

    # Optional in-CSV grouping (column: groups or group)
    groups = split_field(row.get("groups") or row.get("group"))
    for gname in groups:
        gid = get_or_create_group(conn, gname)
        conn.execute(
            "INSERT OR IGNORE INTO trope_group_member(trope_id, group_id) VALUES (?, ?)",
            (tid, gid),
        )

# ---------------------------------- CLI ----------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Load/refresh trope catalog from CSV (supports optional groups, tvtropes_url, anti_aliases, tags)."
    )
    ap.add_argument("--db", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--clear", action="store_true", help="Delete all rows from trope before loading")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    conn = sqlite3.connect(args.db)
    ensure_schema(conn)

    if args.clear:
        # Clear catalog and membership; keep groups so names/ids remain stable across loads
        conn.executescript("DELETE FROM trope_group_member; DELETE FROM trope;")
        conn.commit()

    inserted = 0
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            upsert_trope(conn, row)
            inserted += 1

    conn.commit()
    conn.close()
    print(f"[load_tropes] upserted {inserted} rows into trope")

if __name__ == "__main__":
    main()
