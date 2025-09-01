import os, sqlite3, pytest

DB = os.getenv("TROPES_DB", "ingester/tropes.db")

def _conn():
    if not os.path.exists(DB):
        pytest.skip(f"db not found: {DB}")
    return sqlite3.connect(DB)

def test_support_selection_ranks_contiguous():
    conn = _conn()
    row = conn.execute("SELECT scene_id, COUNT(*) c FROM support_selection GROUP BY scene_id ORDER BY c DESC LIMIT 1").fetchone()
    if not row: pytest.skip("no support_selection rows")
    scene_id = row[0]
    ranks = [r[0] for r in conn.execute("SELECT rank FROM support_selection WHERE scene_id=? ORDER BY rank", (scene_id,))]
    assert ranks == list(range(1, len(ranks)+1)), f"ranks not contiguous: {ranks}"

def test_trope_sanity_weight_range():
    conn = _conn()
    bad = conn.execute("SELECT scene_id, trope_id, weight FROM trope_sanity WHERE weight < 0.0 OR weight > 1.0").fetchall()
    assert not bad, f"weights out of [0,1]: {bad[:5]}"

def test_evidence_spans_within_scene():
    conn = _conn()
    q = conn.execute("""
      SELECT f.id, s.char_start, s.char_end, f.evidence_start, f.evidence_end
      FROM trope_finding f JOIN scene s ON s.id=f.scene_id
      LIMIT 200
    """).fetchall()
    if not q: pytest.skip("no findings")
    for fid, s0, s1, a, b in q:
        assert s0 <= a <= s1, f"{fid}: start out of bounds"
        assert s0 <= b <= s1, f"{fid}: end out of bounds"
        assert a < b, f"{fid}: empty or inverted span"
