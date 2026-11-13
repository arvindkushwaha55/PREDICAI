"""PREDICAI — maintenance module tests (Part 8 gate)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from maintenance import MaintenanceDB  # noqa: E402
from policy import recommend  # noqa: E402


def fresh_db(tmp: Path) -> MaintenanceDB:
    return MaintenanceDB(tmp / "maint.db")


def test_crud_and_cycles_since_service(tmp_path=None):
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        db = fresh_db(Path(d))
        db.register_machine("ENGINE-001", "Demo")
        assert db.get_machine("ENGINE-001")["display_name"] == "Demo"
        db.add_service_record("ENGINE-001", "2026-10-10", 120, "Major Maintenance")
        db.add_service_record("ENGINE-001", "2026-08-02", 40, "Minor Inspection")
        hist = db.service_history("ENGINE-001")
        assert [h["service_cycle"] for h in hist] == [120, 40]  # newest first
        assert db.cycles_since_service("ENGINE-001", 440) == 320
        assert db.cycles_since_service("UNKNOWN", 440) is None
        db.close()


def test_synthetic_seed_is_flagged():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        db = fresh_db(Path(d))
        db.seed_synthetic_demo()
        hist = db.service_history("ENGINE-001")
        assert len(hist) == 2
        assert all(h["is_synthetic"] == 1 for h in hist)
        db.close()


def test_policy_levels_and_reasons():
    r = recommend(predicted_rul=120, cycles_since_service=50)
    assert r.level == "MONITOR" and "120" in r.reason
    r = recommend(predicted_rul=42, cycles_since_service=100)
    assert r.level == "INSPECTION" and "42" in r.reason
    r = recommend(predicted_rul=20, cycles_since_service=100)
    assert r.level == "REVIEW"
    # long service interval escalates a monitoring case
    r = recommend(predicted_rul=120, cycles_since_service=320)
    assert r.level == "INSPECTION" and "320" in r.reason
    # no model yet -> still answers honestly
    r = recommend(predicted_rul=None, cycles_since_service=None)
    assert r.level == "MONITOR" and "no RUL prediction" in r.reason


def test_recommendation_is_stored():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        db = fresh_db(Path(d))
        db.seed_synthetic_demo()
        rec = db.make_recommendation("ENGINE-001", predicted_rul=38.0, current_cycle=440)
        assert rec.level == "REVIEW"  # RUL 38 -> INSPECTION, + 320 cycles since service -> REVIEW
        rows = db.conn.execute("SELECT * FROM recommendations").fetchall()
        assert len(rows) == 1 and rows[0]["reason"] == rec.reason
        db.close()
