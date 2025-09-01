#!/usr/bin/env python3
"""
verifier_pass.py
----------------
Lightweight verifier after the judge step.

For each finding, look at a small window around the evidence span and:
  - flag simple NEGATION cues (no/never/without/lack of …),
  - flag META cues (parody/satire/lampshade/deconstruct/cliché),
  - flag ANTI-* cases like "anti–whodunit", "anti whodunit" using the trope's name/aliases.

Modes:
  --mode flag-only   -> only set verifier_flag
  --mode downweight  -> multiply confidence by factors and set verifier_flag
  --mode delete      -> delete flagged findings

Usage:
  python scripts/verifier_pass.py --db tropes.db --work-id <WORK_ID> \
    --mode downweight --window 40 \
    --neg-downweight 0.6 --meta-downweight 0.75 --antialias-downweight 0.5
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
from typing import List, Tuple

# Unicode dash class (ASCII hyphen + unicode dashes)
DASH_CLS = r"[-\u2010-\u2015]"

# Strong negation cues (global)
NEG_STRONG_RE = re.compile(
    r"\b(?:no|never|without|lack(?:ing)?(?:\s+of)?|absence(?:\s+of)?|free\s+of)\b",
    re.I,
)

# Plain "not" (only counts if very near an alias mention)
NOT_RE = re.compile(r"\bnot\b", re.I)

META_RE = re.compile(
    r"\b(?:parody|satire|meta|lampshade(?:d|s|ing)?|deconstruct(?:ion|ing)?|clich(?:e|é)s?)\b",
    re.I,
)

def ensure_columns(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cols = {r[1] for r in cur.execute("PRAGMA table_info(trope_finding);")}
    if "verifier_flag" not in cols:
        cur.execute("ALTER TABLE trope_finding ADD COLUMN verifier_flag TEXT;")
        conn.commit()

def fetch_findings(conn: sqlite3.Connection, work_id: str) -> List[sqlite3.Row]:
    q = """
    SELECT f.id, f.work_id, f.scene_id, f.trope_id, f.evidence_start, f.evidence_end,
           f.confidence, COALESCE(f.verifier_flag,'') AS verifier_flag,
           w.norm_text, t.name AS trope_name, COALESCE(t.aliases,'') AS aliases_json
    FROM trope_finding f
    JOIN work  w ON w.id = f.work_id
    JOIN trope t ON t.id = f.trope_id
    WHERE f.work_id = ?
    ORDER BY f.created_at ASC
    """
    conn.row_factory = sqlite3.Row
    return conn.execute(q, (work_id,)).fetchall()

# ---- alias helpers (mirrors seeding heuristics) ----
def _norm_alias(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s.strip(",.;:!?\"'()[]{}")

def _escape_token(token: str) -> str:
    esc = re.escape(token)
    esc = (esc.replace("-", DASH_CLS)
              .replace("–", DASH_CLS)
              .replace("—", DASH_CLS)
              .replace(r"\-", DASH_CLS))
    esc = esc.replace("'", r"['\u2019]").replace("’", r"['\u2019]")
    return esc

def build_alias_pattern(alias: str) -> re.Pattern:
    parts = re.split(r"\s+", alias.strip())
    esc = [_escape_token(p) for p in parts if p]
    if not esc:
        return re.compile(r"(?!)")
    if len(esc) == 1 and re.fullmatch(r"[A-Za-z]+", parts[0]):
        core = rf"{esc[0]}(?:s|es)?"
    else:
        joiner = rf"(?:{DASH_CLS}\s*|\s+)"
        core = joiner.join(esc)
        last_src = parts[-1]
        if re.fullmatch(r"[A-Za-z]+", last_src):
            esc_last_plural = rf"(?:{esc[-1]}(?:s|es)?)"
            core = joiner.join([*esc[:-1], esc_last_plural])
    return re.compile(rf"(?<!\w){core}(?!\w)", re.I)

def aliases_for_trope(trope_name: str, aliases_json: str) -> Tuple[List[re.Pattern], str]:
    aliases = []
    if aliases_json:
        try:
            arr = json.loads(aliases_json)
            if isinstance(arr, list):
                aliases.extend([a for a in arr if isinstance(a, str)])
        except Exception:
            pass
    all_norm = list(dict.fromkeys([_norm_alias(trope_name)] + [_norm_alias(a) for a in aliases if a]))
    pats = [build_alias_pattern(a) for a in all_norm]
    # fallback simple string to avoid recompiling if needed
    return pats, (all_norm[0] if all_norm else "")

# ---- flagging logic ----
def has_meta(text: str) -> bool:
    return bool(META_RE.search(text))

def has_anti_alias(text: str, alias_pats: List[re.Pattern]) -> bool:
    if "anti" not in text.lower():
        return False
    for pat in alias_pats:
        for m in pat.finditer(text):
            left = text[max(0, m.start() - 16): m.start()]
            if re.search(rf"(?i)(?<!\w)anti(?:{DASH_CLS}\s*|\s+)$", left):
                return True
    return False

def has_negation(text: str, alias_pats: List[re.Pattern]) -> bool:
    # Strong cues anywhere in the window
    if NEG_STRONG_RE.search(text):
        return True
    # Plain 'not' only counts if very near the alias
    if "not" not in text.lower():
        return False
    for pat in alias_pats:
        for m in pat.finditer(text):
            left  = text[max(0, m.start()-16): m.start()]
            right = text[m.end(): m.end()+16]
            if NOT_RE.search(left) or NOT_RE.search(right):
                return True
    return False

def main():
    ap = argparse.ArgumentParser(description="Negation/meta/anti-* verifier pass.")
    ap.add_argument("--db", required=True)
    ap.add_argument("--work-id", required=True)
    ap.add_argument("--mode", choices=["flag-only", "downweight", "delete"], default="downweight")
    ap.add_argument("--window", type=int, default=40, help="chars around evidence to inspect")
    ap.add_argument("--neg-downweight", type=float, default=0.6)
    ap.add_argument("--meta-downweight", type=float, default=0.75)
    ap.add_argument("--antialias-downweight", type=float, default=0.5)
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    ensure_columns(conn)

    rows = fetch_findings(conn, args.work_id)
    if not rows:
        print("[info] no findings for this work; nothing to verify")
        return

    to_delete, updates, flags_only = [], [], []
    for r in rows:
        fid = r["id"]
        e0 = int(r["evidence_start"] or 0)
        e1 = int(r["evidence_end"] or 0)
        txt = r["norm_text"] or ""
        n   = len(txt)
        e0 = max(0, min(e0, n))
        e1 = max(0, min(e1, n))
        if e1 < e0:
            e0, e1 = e1, e0
        w0 = max(0, e0 - args.window)
        w1 = min(n, e1 + args.window)
        window = txt[w0:w1]

        alias_pats, _ = aliases_for_trope(r["trope_name"] or "", r["aliases_json"] or "")

        neg  = has_negation(window, alias_pats)
        meta = has_meta(window)
        anti = has_anti_alias(window, alias_pats)

        flag = ""
        if neg and anti:
            flag = "negation_anti"
        elif neg:
            flag = "negation_cue"
        elif anti:
            flag = "anti_alias"
        elif meta:
            flag = "meta_cue"

        if not flag:
            continue

        if args.mode == "delete":
            to_delete.append((fid,))
        elif args.mode == "flag-only":
            flags_only.append((flag, fid))
        else:
            conf = float(r["confidence"] or 0.0)
            factor = 1.0
            if neg:  factor *= float(args.neg_downweight)
            if meta: factor *= float(args.meta_downweight)
            if anti: factor *= float(args.antialias_downweight)
            new_conf = max(0.0, min(1.0, conf * factor))
            updates.append((new_conf, flag, fid))

    cur = conn.cursor()
    deleted = updated = flagged = 0
    if to_delete:
        cur.executemany("DELETE FROM trope_finding WHERE id=?", to_delete)
        deleted = cur.rowcount
    if updates:
        cur.executemany("UPDATE trope_finding SET confidence=?, verifier_flag=? WHERE id=?", updates)
        updated = cur.rowcount
    if flags_only:
        cur.executemany("UPDATE trope_finding SET verifier_flag=? WHERE id=?", flags_only)
        flagged = cur.rowcount
    conn.commit()

    print(f"[verifier] mode={args.mode}  updated={updated}  flagged_only={flagged}  deleted={deleted}")

if __name__ == "__main__":
    main()
