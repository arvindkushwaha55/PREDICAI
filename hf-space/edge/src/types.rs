//! Request/response DTOs + validation for the Phase D API (plan.md §3).
//!
//! Contract (matches the Kaggle export + `preprocessing.json`):
//! - window: exactly 30 rows (cycles) x 14 features, in `preprocessing.json`
//!   feature order — the server never reorders or renames features.
//! - RUL is returned in **cycles**, never hours (plan rule 2).

use serde::{Deserialize, Serialize};

/// POST /predict_rul request body.
#[derive(Debug, Clone, Deserialize)]
pub struct PredictRequest {
    pub machine_id: String,
    /// Raw (unnormalized) sensor window: [seq_len][num_features].
    /// f64 on purpose: values stay f64 through normalization and are cast
    /// to f32 once at tensor build, matching the Python pipeline (plan §12).
    pub window: Vec<Vec<f64>>,
}

impl PredictRequest {
    /// Validate shape + finiteness against the preprocessing contract.
    /// Returns a human-readable error; the route turns it into 400 JSON.
    pub fn validate(&self, seq_len: usize, num_features: usize) -> Result<(), String> {
        if self.machine_id.trim().is_empty() {
            return Err("machine_id must not be empty".to_string());
        }
        if self.window.len() != seq_len {
            return Err(format!(
                "window must have exactly {seq_len} rows (cycles), got {}",
                self.window.len()
            ));
        }
        for (i, row) in self.window.iter().enumerate() {
            if row.len() != num_features {
                return Err(format!(
                    "window row {i} must have exactly {num_features} features, got {}",
                    row.len()
                ));
            }
            if row.iter().any(|v| !v.is_finite()) {
                return Err(format!("window row {i} contains NaN or infinite values"));
            }
        }
        Ok(())
    }
}

/// POST /predict_rul response body.
#[derive(Debug, Clone, Serialize)]
pub struct PredictResponse {
    pub machine_id: String,
    pub predicted_rul: f32,
    pub unit: String, // always "cycles"
    pub status: String, // "ok"
}

/// GET /health response body.
#[derive(Debug, Clone, Serialize)]
pub struct HealthResponse {
    pub status: String, // "ok" — the server is up (model may still be missing)
    pub model_loaded: bool,
    pub seq_len: usize,
    pub num_features: usize,
    pub model_path: Option<String>,
}

/// Structured API error body (all errors are JSON, never a bare 500 page).
#[derive(Debug, Clone, Serialize)]
pub struct ErrorResponse {
    pub error: String,
    pub detail: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn good_window() -> Vec<Vec<f64>> {
        vec![vec![0.0; 14]; 30]
    }

    #[test]
    fn valid_window_passes() {
        let req = PredictRequest { machine_id: "ENGINE-001".into(), window: good_window() };
        assert!(req.validate(30, 14).is_ok());
    }

    #[test]
    fn wrong_seq_len_rejected() {
        let req = PredictRequest { machine_id: "ENGINE-001".into(), window: vec![vec![0.0; 14]; 29] };
        assert!(req.validate(30, 14).unwrap_err().contains("30 rows"));
    }

    #[test]
    fn wrong_feature_count_rejected() {
        let req = PredictRequest { machine_id: "ENGINE-001".into(), window: vec![vec![0.0; 13]; 30] };
        assert!(req.validate(30, 14).unwrap_err().contains("14 features"));
    }

    #[test]
    fn nan_rejected() {
        let mut w = good_window();
        w[5][3] = f64::NAN;
        let req = PredictRequest { machine_id: "ENGINE-001".into(), window: w };
        assert!(req.validate(30, 14).unwrap_err().contains("NaN"));
    }

    #[test]
    fn empty_machine_id_rejected() {
        let req = PredictRequest { machine_id: "  ".into(), window: good_window() };
        assert!(req.validate(30, 14).is_err());
    }
}
