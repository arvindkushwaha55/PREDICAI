//! ONNX inference, isolated in this one file (plan Risk R-3).
//!
//! The Kaggle export contract (see `ml/scripts/make_kaggle_notebook.py`):
//! - file: `cmapss_lstm.onnx`, opset 17
//! - input name `"input"`, shape `(batch, 30, 14)` — batch is dynamic
//! - output name `"rul"`, shape `(batch,)` — predicted RUL in cycles
//! - model output is clipped at >= 0 at inference time, matching
//!   `ml/src/predicai_ml/model.py::predict_rul`.
//!
//! Until the evening Kaggle run lands, no ONNX file exists in this repo:
//! the server starts with `model_loaded = false` and `/predict_rul`
//! answers 503 `model_not_loaded` instead of pretending to predict.
//! The parity test in `tests/` is therefore gated on the artifact's
//! presence and is skipped (loudly) until it arrives.

use ort::session::{builder::GraphOptimizationLevel, Session};
use std::path::{Path, PathBuf};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ModelError {
    #[error("ONNX model file not found: {0}")]
    NotFound(PathBuf),
    #[error("failed to load ONNX model: {0}")]
    Load(String),
    #[error("inference failed: {0}")]
    Inference(String),
    #[error("unexpected ONNX output shape {0:?}, expected [1] (one RUL value)")]
    BadOutputShape(Vec<i64>),
}

/// A loaded ONNX session. `Session::run` takes `&mut self`, so callers
/// hold this behind a `Mutex` (see `main.rs`).
pub struct RulModel {
    session: Session,
    pub path: PathBuf,
}

/// With the `load-dynamic` feature (see Cargo.toml), the ONNX Runtime
/// shared library must be pointed at once via ORT_DYLIB_PATH before the
/// first session is created. Idempotent; a no-op when the env var is unset
/// (e.g. a future build that bundles the runtime again).
fn ort_environment() -> Option<&'static ort::environment::Environment> {
    // rc.15 API: environments are explicit — Session::builder(&env).
    // init_from(path)?.build() -> Environment; kept alive in a OnceLock.
    static ENV: std::sync::OnceLock<Option<ort::environment::Environment>> =
        std::sync::OnceLock::new();
    ENV.get_or_init(|| {
        let p = std::env::var("ORT_DYLIB_PATH").ok()?;
        let builder = match ort::init_from(&p) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("ort init_from({p}) failed: {e}");
                return None;
            }
        };
        match builder.build() {
            Ok(env) => Some(env),
            Err(e) => {
                eprintln!("ort environment build failed: {e}");
                None
            }
        }
    })
    .as_ref()
}

impl RulModel {
    pub fn load(path: &Path) -> Result<Self, ModelError> {
        if !path.exists() {
            return Err(ModelError::NotFound(path.to_path_buf()));
        }
        let env = ort_environment().ok_or_else(|| {
            ModelError::Load(
                "ONNX Runtime environment unavailable — set ORT_DYLIB_PATH to libonnxruntime (load-dynamic build)".to_string(),
            )
        })?;
        let session = Session::builder(env)
            .map_err(|e| ModelError::Load(e.to_string()))?
            .with_optimization_level(GraphOptimizationLevel::Level1)
            .map_err(|e| ModelError::Load(e.to_string()))?
            .with_intra_threads(1)
            .map_err(|e| ModelError::Load(e.to_string()))?
            .commit_from_file(path)
            .map_err(|e| ModelError::Load(e.to_string()))?;
        Ok(Self { session, path: path.to_path_buf() })
    }

    /// Run one normalized window (flat, row-major, length 30*14) and
    /// return the predicted RUL in cycles, clipped at >= 0.
    pub fn predict(&mut self, normalized: &[f32], seq_len: usize, num_features: usize) -> Result<f32, ModelError> {
        let expected = seq_len * num_features;
        if normalized.len() != expected {
            return Err(ModelError::Inference(format!(
                "normalized window has {} values, expected {expected}",
                normalized.len()
            )));
        }
        // rc.15: tensors are built with Tensor::from_array((shape, data)).
        let input = ort::value::Tensor::<f32>::from_array((
            vec![1_i64, seq_len as i64, num_features as i64],
            normalized.to_vec(),
        ))
        .map_err(|e| ModelError::Inference(e.to_string()))?;
        let outputs = self
            .session
            .run(ort::inputs!["input" => input])
            .map_err(|e| ModelError::Inference(e.to_string()))?;
        let (shape, data) = outputs["rul"]
            .try_extract_tensor::<f32>()
            .map_err(|e| ModelError::Inference(e.to_string()))?;
        let dims: Vec<i64> = shape.iter().copied().collect();
        if dims != vec![1] {
            return Err(ModelError::BadOutputShape(dims));
        }
        Ok(data[0].max(0.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_model_file_is_a_clear_error() {
        let err = RulModel::load(Path::new("/nonexistent/cmapss_lstm.onnx"))
            .err()
            .expect("loading a missing model file must fail");
        assert!(matches!(err, ModelError::NotFound(_)));
        assert!(err.to_string().contains("not found"));
    }
}
