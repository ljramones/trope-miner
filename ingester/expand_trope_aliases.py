#!/usr/bin/env python3
"""
Expand trope.aliases using a local Ollama model (e.g., qwen2.5:7b-instruct, llama3.1:8b).

- Reads (id, name, best-available description, aliases) from SQLite.
- Calls Ollama /api/chat to generate SHORT surface-form aliases/nicknames.
- Cleans, dedupes, merges with existing aliases, and writes JSON array to trope.aliases.
- Creates a DB backup before first write (unless --dry-run).
- Options: --only-missing, --limit N, --model, --temperature, --dry-run.

Usage:
  export OLLAMA_BASE_URL=http://localhost:11434
  python expand_trope_aliases.py --only-missing --limit 25
  python expand_trope_aliases.py --model llama3.1:8b
"""

import argparse
import json
import os
import re
import shutil
import sqlite3
import sys
import unicodedata
from datetime import datetime
from typing import List, Optional

import requests

OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

SYS_PROMPT = """You generate SHORT, realistic alias strings for trope names.
Goal: maximize literal text hits in prose (gazetteer).
Rules:
- Output ONLY a JSON array of strings (no commentary).
- 1–12 items max. Each item 1–5 words.
- Include common nicknames, surface paraphrases, and near-synonyms.
- Avoid long definitions, punctuation-heavy phrases, quotes, or slurs.
- No duplicates. No numbered lists. No explanations.
- Prefer lowercase; ASCII where possible.
Example: for "Deus Ex Machina" -> ["god from the machine","out of nowhere rescue","sudden rescue"]
"""

USER_PROMPT_TMPL = """Trope:
name: {name}
description: {desc}

Return ONLY JSON array of alias strings (lowercase), diverse surface forms people actually write in stories. 1–12 items.
"""

# ------------------------- Ollama client -------------------------

def _extract_json_array(text: str) -> List[str]:
    """Best-effort extraction of a JSON array from free-form LLM output."""
    s = text.strip()
    # code-fence block?
    fence = re.search(r"```(?:json)?\s*(\[[\s\S]*?\])\s*```", s, re.IGNORECASE)
    if fence:
        s = fence.group(1)
    if not s.startswith("["):
        br = re.search(r"\[[\s\S]*\]", s)
        if br:
            s = br.group(0)
    try:
        obj = json.loads(s)
        if isinstance(obj, list):
            return [x for x in obj if isinstance(x, str)]
    except Exception:
        pass
    return []

def call_ollama_chat(model: str, name: str, desc: Optional[str],
                     temperature: float = 0.2, timeout: int = 90) -> List[str]:
    """Call local Ollama /api/chat and parse a JSON array of strings."""
    url = f"{OLLAMA_URL.rstrip('/')}/api/chat"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": USER_PROMPT_TMPL.format(name=name, desc=(desc or "")[:800])}
        ],
        "stream": False,
        "options": {"temperature": temperature}
    }
    try:
        r = requests.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        content = (r.json() or {}).get("message", {}).get("content", "") or ""
        return _extract_json_array(content)
    except requests.RequestException as e:
        print(f"[warn] Ollama request failed for {name!r}: {e}", file=sys.stderr)
    except ValueError as e:
        print(f"[warn] JSON parse failed for {name!r}: {e}", file=sys.stderr)
    return []

# ------------------------- Normalization -------------------------

def ascii_fold(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")

def normalize_alias(s: str) -> str:
    s = s.strip().lower()
    s = ascii_fold(s)
    s = re.sub(r"\s+", " ", s)
    s = s.strip(",.;:!?\"'()[]{}")
    if len(s) > 40:
        s = s[:40].rstrip()
    return s

def clean_aliases(candidates: List[str], canonical_name: str) -> List[str]:
    """Normalize, filter junk, drop dupes, cap at 12. Never return the canonical name."""
    out: List[str] = []
    seen = set()
    canon_norm = normalize_alias(canonical_name)
    for a in candidates:
        n = normalize_alias(a)
        if not n or n == canon_norm or len(n) < 2 or n in seen:
            continue
        if n in {"trope", "cliche", "stereotype"}:
            continue
        if any(ch in n for ch in ['"', "'", "`"]):
            continue
        out.append(n)
        seen.add(n)
        if len(out) >= 12:
            break
    return out

def merge_aliases(existing_json: Optional[str], new_aliases: List[str]) -> List[str]:
    existing: List[str] = []
    if existing_json:
        try:
            data = json.loads(existing_json)
            if isinstance(data, list):
                existing = [normalize_alias(x) for x in data if isinstance(x, str)]
        except Exception:
            pass
    merged, seen = [], set()
    for a in existing + new_aliases:
        if a and a not in seen:
            merged.append(a)
            seen.add(a)
    return merged

# ------------------------- SQLite helpers -------------------------

def get_table_columns(conn: sqlite3.Connection, table: str) -> set:
    rows = conn.execute(f"PRAGMA table_info({table});").fetchall()
    return {r[1] for r in rows}  # column name at index 1

def ensure_aliases_column(conn: sqlite3.Connection) -> None:
    rows = conn.execute("PRAGMA table_info(trope);").fetchall()
    if not any(row[1] == "aliases" for row in rows):
        conn.execute("ALTER TABLE trope ADD COLUMN aliases TEXT;")
        conn.commit()

def fetch_tropes(conn: sqlite3.Connection, only_missing: bool, limit: Optional[int]):
    cols = get_table_columns(conn, "trope")

    # Pick a description-like column if present; always alias to "description"
    desc_col = None
    for candidate in ("description", "definition", "summary", "text", "details", "body"):
        if candidate in cols:
            desc_col = candidate
            break
    select_desc = f"{desc_col} AS description" if desc_col else "NULL AS description"

    aliases_in_table = "aliases" in cols
    select_aliases = "aliases" if aliases_in_table else "NULL AS aliases"

    sql = f"SELECT id, name, {select_desc}, {select_aliases} FROM trope"
    if only_missing and aliases_in_table:
        sql += " WHERE aliases IS NULL OR TRIM(aliases) = ''"
    sql += " ORDER BY name COLLATE NOCASE"
    if limit:
        sql += f" LIMIT {int(limit)}"

    return conn.execute(sql).fetchall()

def ensure_backup(db_path: str) -> str:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    bak = f"{db_path}.bak-{ts}"
    shutil.copyfile(db_path, bak)
    return bak

# ------------------------- Main -------------------------

def main():
    ap = argparse.ArgumentParser(description="Expand trope.aliases via local Ollama (Qwen/LLaMA).")
    ap.add_argument("--db", default="./tropes.db", help="Path to SQLite DB (default: ./tropes.db)")
    ap.add_argument("--model", default="qwen2.5:7b-instruct",
                    help="Ollama model, e.g. qwen2.5:7b-instruct or llama3.1:8b")
    ap.add_argument("--only-missing", action="store_true",
                    help="Process only rows where aliases is NULL/empty (if column exists)")
    ap.add_argument("--limit", type=int, default=None, help="Max number of tropes to process")
    ap.add_argument("--dry-run", action="store_true", help="Do not write changes to DB")
    ap.add_argument("--temperature", type=float, default=0.2, help="LLM temperature (default 0.2)")
    args = ap.parse_args()

    if not os.path.exists(args.db):
        print(f"DB not found: {args.db}", file=sys.stderr)
        sys.exit(1)

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row

    # Ensure aliases column exists (safe if already present)
    ensure_aliases_column(conn)

    rows = fetch_tropes(conn, args.only_missing, args.limit)
    if not rows:
        print("No tropes matched selection.")
        conn.close()
        return

    if not args.dry_run:
        bak = ensure_backup(args.db)
        print(f"Backup created: {bak}")

    cur = conn.cursor()
    updated = 0

    for row in rows:
        tid = row["id"]
        name = row["name"]
        desc = row["description"]  # guaranteed alias name in SELECT (may be None)
        existing_aliases_json = row["aliases"]

        llm_out = call_ollama_chat(
            model=args.model,
            name=name,
            desc=desc,
            temperature=args.temperature
        )
        cleaned = clean_aliases(llm_out, canonical_name=name)
        merged = merge_aliases(existing_aliases_json, cleaned)

        print(f"- {name}: +{len(cleaned)} (total {len(merged)})")
        if args.dry_run:
            continue

        cur.execute(
            "UPDATE trope SET aliases=? WHERE id=?",
            (json.dumps(merged, ensure_ascii=True), tid)
        )
        updated += 1

    if not args.dry_run:
        conn.commit()
        print(f"\nUpdated aliases for {updated} tropes.")
    else:
        print("\nDry run complete. No changes written.")

    conn.close()

if __name__ == "__main__":
    main()
