-- PREDICAI — Maintenance History schema (Part 8 / Phase B)
-- One schema, two consumers: the Python reference module now, the Rust
-- edge engine later. FD001 contains NO real maintenance records — every
-- demo row is synthetic and must carry is_synthetic = 1.

CREATE TABLE IF NOT EXISTS machines (
    machine_id   TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    registered_at TEXT NOT NULL DEFAULT (datetime('now')),
    notes        TEXT
);

CREATE TABLE IF NOT EXISTS service_records (
    record_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    machine_id    TEXT NOT NULL REFERENCES machines(machine_id),
    service_date  TEXT NOT NULL,             -- ISO date, e.g. 2026-10-10
    service_cycle INTEGER NOT NULL,          -- machine cycle at service time
    service_type  TEXT NOT NULL,             -- e.g. Minor Inspection / Major Maintenance
    notes         TEXT,
    is_synthetic  INTEGER NOT NULL DEFAULT 0 CHECK (is_synthetic IN (0, 1)),
    created_at    TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS recommendations (
    rec_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    machine_id      TEXT NOT NULL REFERENCES machines(machine_id),
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    predicted_rul   REAL,                    -- cycles, from the LSTM (nullable pre-model)
    current_cycle   INTEGER,
    cycles_since_service INTEGER,
    level           TEXT NOT NULL,           -- MONITOR / INSPECTION / REVIEW
    reason          TEXT NOT NULL,           -- human-readable, always present
    policy_version  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS predictions_log (
    pred_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    machine_id    TEXT NOT NULL,
    created_at    TEXT NOT NULL DEFAULT (datetime('now')),
    at_cycle      INTEGER,
    predicted_rul REAL NOT NULL,
    unit          TEXT NOT NULL DEFAULT 'cycles',
    model_version TEXT,
    source        TEXT NOT NULL DEFAULT 'SIMULATED REPLAY'
);
