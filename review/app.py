#!/usr/bin/env python3
from __future__ import annotations
import os, sqlite3, uuid
from pathlib import Path
from flask import Flask, g, render_template, request, jsonify, abort

# --- Paths & config -------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DEFAULT_DB = (ROOT.parent / "ingester" / "tropes.db").as_posix()
DB_PATH = os.getenv("TROPES_DB", DEFAULT_DB)

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config.update(
    SECRET_KEY=os.getenv("REVIEW_SECRET", "dev"),
    # In dev, never cache static assets (so updated JS/CSS always loads)
    SEND_FILE_MAX_AGE_DEFAULT=0,
)

# ---- Display-time quote fixes (doesn't affect DB offsets) ----------------
DISPLAY_CHAR_MAP = str.maketrans({
    # Some MacRoman-encoded punctuation we observed in text dumps
    "Ò": "“", "Ó": "”", "Õ": "’", "Ô": "—", "Ê": "—",
})
def display_fix_quotes(s: str) -> str:
    return s.translate(DISPLAY_CHAR_MAP)

# --- DB helpers -----------------------------------------------------------
def get_db() -> sqlite3.Connection:
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db

@app.teardown_appcontext
def close_db(_exc):
    db = g.pop("db", None)
    if db is not None:
        db.close()

def ensure_review_schema(conn: sqlite3.Connection) -> None:
    """
    Creates the human review table + view if missing.
    """
    conn.executescript("""
    PRAGMA foreign_keys=ON;

    CREATE TABLE IF NOT EXISTS trope_finding_human (
      id                 TEXT PRIMARY KEY,
      finding_id         TEXT NOT NULL,
      decision           TEXT NOT NULL CHECK (decision IN ('accept','reject','edit')),
      corrected_start    INTEGER,
      corrected_end      INTEGER,
      corrected_trope_id TEXT,
      note               TEXT,
      reviewer           TEXT,
      created_at         TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
      FOREIGN KEY(finding_id) REFERENCES trope_finding(id) ON DELETE CASCADE
    );
    CREATE INDEX IF NOT EXISTS idx_tfh_finding ON trope_finding_human(finding_id);

    CREATE VIEW IF NOT EXISTS v_latest_human AS
    SELECT h.*
    FROM trope_finding_human h
    JOIN (
      SELECT finding_id, MAX(created_at) AS mx
      FROM trope_finding_human
      GROUP BY finding_id
    ) last
      ON last.finding_id = h.finding_id AND last.mx = h.created_at;
    """)
    conn.commit()

# --- Pages ----------------------------------------------------------------
@app.route("/")
def index():
    db = get_db()
    ensure_review_schema(db)
    works = db.execute("""
      SELECT w.id,
             w.title,
             w.author,
             COALESCE(tf.cnt, 0) AS findings,
             COALESCE(hc.cnt, 0) AS reviewed
      FROM work w
      LEFT JOIN (
        SELECT work_id, COUNT(*) AS cnt
        FROM trope_finding
        GROUP BY work_id
      ) tf ON tf.work_id = w.id
      LEFT JOIN (
        SELECT f.work_id, COUNT(DISTINCT h.finding_id) AS cnt
        FROM trope_finding f
        JOIN v_latest_human h ON h.finding_id = f.id
        GROUP BY f.work_id
      ) hc ON hc.work_id = w.id
      ORDER BY w.created_at DESC
    """).fetchall()
    return render_template("index.html", works=works)

@app.route("/work/<work_id>")
def work(work_id: str):
    db = get_db(); ensure_review_schema(db)
    ensure_review_schema(db)  # <-- ensure the view exists on deep links
    scenes = db.execute("""
      SELECT s.id,
             s.idx,
             (s.char_end - s.char_start) AS len,
             COALESCE(tf.cnt, 0)         AS findings,
             COALESCE(acc.cnt, 0)        AS accepted,
             COALESCE(rej.cnt, 0)        AS rejected
      FROM scene s
      LEFT JOIN (
        SELECT scene_id, COUNT(*) AS cnt
        FROM trope_finding
        GROUP BY scene_id
      ) tf ON tf.scene_id = s.id
      LEFT JOIN (
        SELECT f.scene_id, COUNT(*) AS cnt
        FROM trope_finding f
        JOIN v_latest_human h ON h.finding_id = f.id AND h.decision = 'accept'
        GROUP BY f.scene_id
      ) acc ON acc.scene_id = s.id
      LEFT JOIN (
        SELECT f.scene_id, COUNT(*) AS cnt
        FROM trope_finding f
        JOIN v_latest_human h ON h.finding_id = f.id AND h.decision = 'reject'
        GROUP BY f.scene_id
      ) rej ON rej.scene_id = s.id
      WHERE s.work_id = ?
      ORDER BY s.idx
    """, (work_id,)).fetchall()
    w = db.execute("SELECT id, title, author FROM work WHERE id = ?", (work_id,)).fetchone()
    return render_template("index.html", works=[], scenes=scenes, current_work=w)

@app.route("/scene/<scene_id>")
def scene(scene_id: str):
    db = get_db();
    ensure_review_schema(db)
    ensure_review_schema(db)  # <-- ensure the view exists on deep links
    row = db.execute("""
      SELECT s.id, s.idx, s.work_id, s.char_start, s.char_end,
             w.title, w.author, w.norm_text
      FROM scene s
      JOIN work w ON w.id = s.work_id
      WHERE s.id = ?
    """, (scene_id,)).fetchone()
    if not row:
        abort(404)

    s0, s1 = int(row["char_start"]), int(row["char_end"])
    full = row["norm_text"] or ""
    # Note: display_fix_quotes does 1:1 character replacements (same length)
    scene_text = display_fix_quotes(full[s0:s1])

    findings = db.execute("""
      SELECT f.id,
             f.trope_id,
             t.name AS trope,
             f.confidence,
             f.evidence_start AS start,
             f.evidence_end   AS end,
             f.rationale,
             h.decision
      FROM trope_finding f
      JOIN trope t ON t.id = f.trope_id
      LEFT JOIN v_latest_human h ON h.finding_id = f.id
      WHERE f.scene_id = ?
      ORDER BY f.evidence_start, f.evidence_end
    """, (scene_id,)).fetchall()

    # Spans JSON for the highlighter script
    spans = []
    for r in findings:
        try:
            spans.append({
                "id": r["id"],
                "start": int(r["start"]),
                "end": int(r["end"]),
                "trope": r["trope"],
                "confidence": float(r["confidence"] or 0.0),
            })
        except Exception:
            # Skip any malformed row rather than crashing
            continue

    tropes = db.execute("SELECT id, name FROM trope ORDER BY name COLLATE NOCASE").fetchall()

    return render_template(
        "scene.html",
        work_id=row["work_id"],
        title=row["title"],
        author=row["author"],
        scene_id=row["id"],
        scene_idx=row["idx"],
        scene_text=scene_text,
        offset=s0,
        findings=findings,
        tropes=tropes,
        spans=spans,
    )

# --- API: accept / reject / edit / new -----------------------------------
def _uuid() -> str:
    return str(uuid.uuid4())

@app.post("/api/decision")
def api_decision():
    data = request.get_json(force=True)
    fid = data.get("finding_id")
    decision = data.get("decision")
    note = (data.get("note") or None)
    reviewer = (data.get("reviewer") or None)
    if decision not in ("accept", "reject"):
        return jsonify({"ok": False, "error": "decision must be accept|reject"}), 400

    db = get_db()
    ensure_review_schema(db)
    db.execute(
        "INSERT INTO trope_finding_human(id, finding_id, decision, note, reviewer) VALUES(?,?,?,?,?)",
        (_uuid(), fid, decision, note, reviewer)
    )
    db.commit()
    return jsonify({"ok": True})

@app.post("/api/edit_span")
def api_edit_span():
    data = request.get_json(force=True)
    fid = data.get("finding_id")
    start = int(data.get("start"))
    end = int(data.get("end"))
    trope_id = data.get("trope_id") or None
    note = data.get("note") or None
    reviewer = data.get("reviewer") or None

    db = get_db()
    ensure_review_schema(db)

    # Look up text length via the finding’s work_id
    row = db.execute("""
      SELECT f.work_id, COALESCE(length(w.norm_text), 0) AS n
      FROM trope_finding f
      JOIN work w ON w.id = f.work_id
      WHERE f.id = ?
    """, (fid,)).fetchone()
    if not row:
        return jsonify({"ok": False, "error": "finding not found"}), 404

    N = int(row["n"])
    start = max(0, min(start, N))
    end   = max(0, min(end,   N))
    if end <= start:
        return jsonify({"ok": False, "error": "end must be > start"}), 400

    # History row
    db.execute("""
      INSERT INTO trope_finding_human
        (id, finding_id, decision, corrected_start, corrected_end, corrected_trope_id, note, reviewer)
      VALUES (?, ?, 'edit', ?, ?, ?, ?, ?)
    """, (_uuid(), fid, start, end, trope_id, note, reviewer))

    # Apply to main finding (non-destructive; history above)
    if trope_id:
        db.execute(
            "UPDATE trope_finding SET evidence_start=?, evidence_end=?, trope_id=? WHERE id=?",
            (start, end, trope_id, fid)
        )
    else:
        db.execute(
            "UPDATE trope_finding SET evidence_start=?, evidence_end=? WHERE id=?",
            (start, end, fid)
        )
    db.commit()
    return jsonify({"ok": True})

@app.post("/api/new_finding")
def api_new_finding():
    data = request.get_json(force=True)
    scene_id = data["scene_id"]; work_id = data["work_id"]; trope_id = data["trope_id"]
    start = int(data["start"]);   end     = int(data["end"])
    conf = float(data.get("confidence", 0.7))
    rationale = data.get("rationale") or ""

    db = get_db()
    Nrow = db.execute("SELECT COALESCE(length(norm_text),0) AS n FROM work WHERE id=?", (work_id,)).fetchone()
    N = int(Nrow["n"]) if Nrow else 0
    start = max(0, min(start, N))
    end   = max(0, min(end,   N))
    if end <= start:
        return jsonify({"ok": False, "error": "end must be > start"}), 400

    fid = _uuid()
    try:
        db.execute("""
          INSERT OR IGNORE INTO trope_finding
            (id, work_id, scene_id, chunk_id, trope_id, level, confidence, rationale, evidence_start, evidence_end, model)
          VALUES (?, ?, ?, NULL, ?, 'span', ?, ?, ?, ?, 'human')
        """, (fid, work_id, scene_id, trope_id, conf, rationale, start, end))
        db.commit()
    except sqlite3.IntegrityError:
        return jsonify({"ok": False, "error": "duplicate finding"}), 409

    return jsonify({"ok": True, "id": fid})

# --- Diagnostics ----------------------------------------------------------
@app.get("/__diag")
def diag():
    db = get_db()
    c = db.cursor()
    w = c.execute("SELECT COUNT(*) FROM work").fetchone()[0]
    s = c.execute("SELECT COUNT(*) FROM scene").fetchone()[0]
    f = c.execute("SELECT COUNT(*) FROM trope_finding").fetchone()[0]
    bad = c.execute("""
      SELECT COUNT(*)
      FROM trope_finding f
      JOIN scene s ON s.id = f.scene_id
      WHERE NOT (f.evidence_start >= s.char_start AND f.evidence_end <= s.char_end)
    """).fetchone()[0]
    return jsonify({"ok": True, "db": DB_PATH, "works": w, "scenes": s, "findings": f, "findings_outside_scene": bad})


@app.get("/scene_plain/<scene_id>")
def scene_plain(scene_id: str):
    db = get_db()
    # make sure the review view exists in all routes that might use it
    ensure_review_schema(db)  # <- harmless if already created  :contentReference[oaicite:1]{index=1}
    row = db.execute("""
      SELECT s.char_start, s.char_end, w.norm_text, w.title, w.author, s.idx
      FROM scene s JOIN work w ON w.id = s.work_id
      WHERE s.id = ?
    """, (scene_id,)).fetchone()
    if not row:
      abort(404)
    s0, s1 = int(row["char_start"]), int(row["char_end"])
    txt = (row["norm_text"] or "")[s0:s1]
    # inline styles so nothing can hide it
    html = f"""
<!doctype html><meta charset="utf-8">
<title>Plain Scene {row['idx']}</title>
<div style="padding:16px;font:16px/1.5 -apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial,sans-serif;color:#111;background:#fff">
  <h2 style="margin:0 0 8px">{escape(row['title'])}</h2>
  <div style="margin:0 0 16px;opacity:.8">{escape(row['author'])} — Scene #{row['idx']}</div>
  <pre style="white-space:pre-wrap;margin:0">{escape(txt)}</pre>
</div>
"""
    return html

@app.get("/healthz")
def healthz():
    return jsonify({"ok": True, "db": DB_PATH})

if __name__ == "__main__":
    print(f"* Using DB: {DB_PATH}")
    app.run(host="127.0.0.1", port=5050, debug=True)
