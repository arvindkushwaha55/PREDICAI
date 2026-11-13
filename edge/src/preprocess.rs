//! Preprocessing parity with Python (plan rule 5 / D-6).
//!
//! The Rust engine never hardcodes scaler statistics: it reads the same
//! `ml/artifacts/preprocessing.json` the Python pipeline wrote
//! (train-only z-score scaler, 14 features in fixed order, seq_len 30,
//! RUL cap 125). Normalization is `(x - mean) / scale` per feature —
//! identical to `ml/src/predicai_ml/preprocessing.py`.
//!
//! Parity is proven by test, not intention: `normalize_matches_python`
//! below recomputes the formula by hand on real scaler values, and the
//! Phase D parity test (once the ONNX file lands) compares Rust vs Python
//! end-to-end on the same fixed windows (tolerance 1e-3).

use serde::Deserialize;
use std::path::Path;

/// The shared Python <-> Rust preprocessing contract.
#[derive(Debug, Clone, Deserialize)]
pub struct Preprocessing {
    pub version: u32,
    pub dataset: String,
    pub features: Vec<String>,
    pub num_features: usize,
    pub seq_len: usize,
    pub rul_cap: u32,
    pub scaler: Scaler,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Scaler {
    #[serde(rename = "type")]
    pub kind: String,
    pub mean: Vec<f64>,
    pub scale: Vec<f64>,
}

impl Preprocessing {
    /// Load + sanity-check the artifact. Fail fast with a clear error if
    /// the file is missing or internally inconsistent (plan Phase D).
    pub fn load(path: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("cannot read preprocessing.json at {}: {e}", path.display()))?;
        let p: Preprocessing = serde_json::from_str(&text)
            .map_err(|e| format!("cannot parse preprocessing.json at {}: {e}", path.display()))?;
        p.check()?;
        Ok(p)
    }

    fn check(&self) -> Result<(), String> {
        if self.features.len() != self.num_features {
            return Err(format!(
                "preprocessing.json: features.len() {} != num_features {}",
                self.features.len(),
                self.num_features
            ));
        }
        if self.scaler.mean.len() != self.num_features || self.scaler.scale.len() != self.num_features {
            return Err(format!(
                "preprocessing.json: scaler mean/scale length must equal num_features {}",
                self.num_features
            ));
        }
        if self.scaler.scale.iter().any(|s| *s == 0.0 || !s.is_finite()) {
            return Err("preprocessing.json: scaler scale contains zero or non-finite values".to_string());
        }
        if self.seq_len == 0 {
            return Err("preprocessing.json: seq_len must be > 0".to_string());
        }
        Ok(())
    }

    /// Normalize one raw window (rows = cycles, cols = features in contract
    /// order) into a flat f32 tensor in row-major order: (1, seq_len, F).
    /// Input values must already be validated finite by the caller.
    ///
    /// Precision rule (plan §12): raw values stay f64 through the z-score
    /// math and are cast to f32 ONCE, at tensor build — exactly like the
    /// Python pipeline (float64 scaler, float32 windows). Casting the raw
    /// inputs to f32 first measurably moves predictions (~5e-3 on RUL);
    /// the parity test in tests/api.rs guards this.
    pub fn normalize(&self, window: &[Vec<f64>]) -> Vec<f32> {
        let mut out = Vec::with_capacity(window.len() * self.num_features);
        for row in window {
            for (j, &v) in row.iter().enumerate() {
                let z = (v - self.scaler.mean[j]) / self.scaler.scale[j];
                out.push(z as f32);
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The real artifact produced by Part 3 — the test reads it, so a
    /// future regeneration that changes the contract breaks loudly here.
    fn real_artifact() -> Preprocessing {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../ml/artifacts/preprocessing.json");
        Preprocessing::load(&path).expect("preprocessing.json must load")
    }

    #[test]
    fn artifact_contract_is_fd001_14x30() {
        let p = real_artifact();
        assert_eq!(p.dataset, "NASA C-MAPSS FD001");
        assert_eq!(p.num_features, 14);
        assert_eq!(p.seq_len, 30);
        assert_eq!(p.rul_cap, 125);
        assert_eq!(p.features[0], "sensor_2");
        assert_eq!(p.features[13], "sensor_21");
    }

    #[test]
    fn normalize_matches_python_formula() {
        let p = real_artifact();
        // Hand-computed with the real scaler values, first feature only:
        // sensor_2 mean 642.6814172477501, scale 0.5000813389855251.
        // x = 643.0 -> (643.0 - 642.6814172477501) / 0.5000813389855251 ≈ 0.6371
        let mut window = vec![vec![0.0f64; 14]; 30];
        window[0][0] = 643.0;
        let norm = p.normalize(&window);
        assert_eq!(norm.len(), 30 * 14);
        let expected = (643.0f64 - 642.6814172477501) / 0.5000813389855251;
        assert!((norm[0] as f64 - expected).abs() < 1e-5, "got {}, want {expected}", norm[0]);
        // A raw value equal to the mean normalizes to (almost) zero.
        window[0][1] = 1590.5250646641634;
        let norm = p.normalize(&window);
        assert!(norm[1].abs() < 1e-3, "mean value should normalize to ~0, got {}", norm[1]);
    }

    #[test]
    fn missing_file_fails_with_clear_message() {
        let err = Preprocessing::load(Path::new("/nonexistent/preprocessing.json"))
            .unwrap_err();
        assert!(err.contains("cannot read preprocessing.json"));
    }
}
