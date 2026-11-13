"""PREDICAI — Maintenance History module (Part 8 / Phase B), SQLite.

Reference implementation used by tests and the simulator; the Rust engine
will read/write the same database file using the same schema.sql.

Rules (from plan.md):
- FD001 has no real service records: demo data is synthetic and flagged.
- Service history is NEVER an LSTM input and a service event NEVER resets
  the model's RUL. This module only tracks history and recommendations.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from policy import POLICY_VERSION, Recommendation, recommend

SCHEMA = Path(__file__).with_name("schema.sql")


class MaintenanceDB:
    def __init__(self, db_path: str | Path) -> None:
        self.conn = sqlite3.connect(str(db_path))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(SCHEMA.read_text())
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    # ------------------------------------------------------------ machines
    def register_machine(self, machine_id: str, display_name: str,
                         notes: str | None = None) -> None:
        self.conn.execute(
            "INSERT INTO machines (machine_id, display_name, notes) VALUES (?, ?, ?)",
            (machine_id, display_name, notes),
        )
        self.conn.commit()

    def get_machine(self, machine_id: str) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM machines WHERE machine_id = ?", (machine_id,)
        ).fetchone()
        return dict(row) if row else None

    # ------------------------------------------------------------- records
    def add_service_record(self, machine_id: str, service_date: str,
                           service_cycle: int, service_type: str,
                           notes: str | None = None,
                           is_synthetic: bool = False) -> int:
        cur = self.conn.execute(
            """INSERT INTO service_records
               (machine_id, service_date, service_cycle, service_type, notes, is_synthetic)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (machine_id, service_date, service_cycle, service_type, notes,
             int(is_synthetic)),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def service_history(self, machine_id: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM service_records WHERE machine_id = ? "
            "ORDER BY service_cycle DESC, record_id DESC",
            (machine_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def last_service(self, machine_id: str) -> dict | None:
        hist = self.service_history(machine_id)
        return hist[0] if hist else None

    def cycles_since_service(self, machine_id: str, current_cycle: int) -> int | None:
        last = self.last_service(machine_id)
        if last is None:
            return None
        return int(current_cycle) - int(last["service_cycle"])

    # ------------------------------------------------------ recommendations
    def make_recommendation(self, machine_id: str, predicted_rul: float | None,
                            current_cycle: int) -> Recommendation:
        last = self.last_service(machine_id)
        css = self.cycles_since_service(machine_id, current_cycle)
        rec = recommend(predicted_rul, css,
                        last["service_type"] if last else None)
        self.conn.execute(
            """INSERT INTO recommendations
               (machine_id, predicted_rul, current_cycle, cycles_since_service,
                level, reason, policy_version)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (machine_id, predicted_rul, current_cycle, css,
             rec.level, rec.reason, POLICY_VERSION),
        )
        self.conn.commit()
        return rec

    # ------------------------------------------------------------ seed demo
    def seed_synthetic_demo(self) -> None:
        """Clearly-labelled SYNTHETIC demo data — not real service history."""
        if self.get_machine("ENGINE-001") is None:
            self.register_machine(
                "ENGINE-001", "Demo Engine 001",
                notes="SYNTHETIC DEMO machine for PREDICAI demonstration.")
        existing = self.service_history("ENGINE-001")
        if not existing:
            self.add_service_record(
                "ENGINE-001", "2026-10-10", 120, "Major Maintenance",
                notes="SYNTHETIC DEMO record — not a real service event.",
                is_synthetic=True)
            self.add_service_record(
                "ENGINE-001", "2026-08-02", 40, "Minor Inspection",
                notes="SYNTHETIC DEMO record — not a real service event.",
                is_synthetic=True)
