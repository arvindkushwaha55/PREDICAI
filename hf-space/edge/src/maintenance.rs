//! SQLite maintenance history for the Rust engine (plan Phase D).
//!
//! Same database, same schema as the Python module (`maintenance/schema.sql`):
//! the Rust engine applies that schema file (via `CREATE TABLE IF NOT
//! EXISTS`) and logs every prediction into `predictions_log` with
//! `source = 'RUST EDGE'` — never as real sensor data, and never as a
//! maintenance record. Service history stays the Python module's job;
//! this file only reads it for the recommendation endpoint.

use rusqlite::{params, Connection};
use serde::Serialize;
use std::path::Path;
use std::sync::Mutex;

/// Minimal read model for GET /machines/{id}/history.
#[derive(Debug, Clone, Serialize)]
pub struct ServiceRecord {
    pub service_date: String,
    pub service_cycle: i64,
    pub service_type: String,
    pub is_synthetic: i64,
}

pub struct Store {
    conn: Mutex<Connection>,
}

impl Store {
    /// Open (creating if needed) the SQLite DB and apply the shared schema.
    pub fn open(path: &Path) -> Result<Self, rusqlite::Error> {
        let conn = Connection::open(path)?;
        conn.execute_batch(include_str!("../../maintenance/schema.sql"))?;
        Ok(Self { conn: Mutex::new(conn) })
    }

    /// Open an in-memory DB (tests + ephemeral runs).
    pub fn open_in_memory() -> Result<Self, rusqlite::Error> {
        let conn = Connection::open_in_memory()?;
        conn.execute_batch(include_str!("../../maintenance/schema.sql"))?;
        Ok(Self { conn: Mutex::new(conn) })
    }

    pub fn log_prediction(
        &self,
        machine_id: &str,
        at_cycle: Option<i64>,
        predicted_rul: f32,
        model_version: &str,
    ) -> Result<(), rusqlite::Error> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO predictions_log (machine_id, at_cycle, predicted_rul, unit, model_version, source)
             VALUES (?1, ?2, ?3, 'cycles', ?4, 'RUST EDGE')",
            params![machine_id, at_cycle, predicted_rul, model_version],
        )?;
        Ok(())
    }

    pub fn service_history(&self, machine_id: &str) -> Result<Vec<ServiceRecord>, rusqlite::Error> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT service_date, service_cycle, service_type, is_synthetic
             FROM service_records WHERE machine_id = ?1 ORDER BY service_cycle",
        )?;
        let rows = stmt.query_map(params![machine_id], |row| {
            Ok(ServiceRecord {
                service_date: row.get(0)?,
                service_cycle: row.get(1)?,
                service_type: row.get(2)?,
                is_synthetic: row.get(3)?,
            })
        })?;
        rows.collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log_and_read_roundtrip() {
        let store = Store::open_in_memory().unwrap();
        store.log_prediction("ENGINE-001", Some(120), 42.5, "kaggle-evening-1").unwrap();
        let conn = store.conn.lock().unwrap();
        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM predictions_log", [], |r| r.get(0))
            .unwrap();
        assert_eq!(count, 1);
        let source: String = conn
            .query_row("SELECT source FROM predictions_log", [], |r| r.get(0))
            .unwrap();
        assert_eq!(source, "RUST EDGE");
    }

    #[test]
    fn empty_history_for_unknown_machine() {
        let store = Store::open_in_memory().unwrap();
        assert!(store.service_history("NOPE").unwrap().is_empty());
    }
}
