//! PREDICAI Rust edge engine — Phase D (plan.md).
//!
//! Startup contract: load `preprocessing.json` (fail fast if missing or
//! inconsistent — the Python/Rust contract is sacred), then try to load
//! `cmapss_lstm.onnx`. A missing ONNX file does NOT stop startup: the
//! server runs with `model_loaded: false` until the evening Kaggle run
//! produces the model. See `routes.rs` for the 503 behaviour.
//!
//! Env (all optional, defaults are repo-relative):
//! - `PREDICAI_PREPROCESSING` — path to preprocessing.json
//! - `PREDICAI_MODEL`         — path to cmapss_lstm.onnx
//! - `PREDICAI_DB`            — path to the SQLite maintenance DB
//! - `PREDICAI_PORT`          — listen port (default 8080)
//!
//! ```sh
//! cargo run --manifest-path edge/Cargo.toml
//! curl localhost:8080/health
//! ```

use predicai_edge::maintenance::Store;
use predicai_edge::model::RulModel;
use predicai_edge::preprocess::Preprocessing;
use predicai_edge::routes::{router, AppState};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

fn default_path(env: &str, fallback: &str) -> PathBuf {
    std::env::var(env).map(PathBuf::from).unwrap_or_else(|_| {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(fallback)
    })
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let preprocessing_path = default_path("PREDICAI_PREPROCESSING", "../ml/artifacts/preprocessing.json");
    let preprocessing = Preprocessing::load(&preprocessing_path)
        .map_err(|e| anyhow::anyhow!(e))?;

    let model_path = default_path("PREDICAI_MODEL", "../ml/artifacts/cmapss_lstm.onnx");
    let model = match RulModel::load(&model_path) {
        Ok(m) => {
            tracing::info!("ONNX model loaded: {}", model_path.display());
            Some(m)
        }
        Err(e) => {
            // Expected until the evening Kaggle run lands — warn, don't die.
            tracing::warn!("ONNX model not loaded ({e}); /health will report model_loaded=false");
            None
        }
    };

    let db_path = default_path("PREDICAI_DB", "../maintenance/predicai.db");
    let store = Store::open(&db_path)?;

    let state = Arc::new(AppState {
        preprocessing,
        model: Mutex::new(model),
        model_path: Some(model_path.display().to_string()),
        store,
    });

    let port: u16 = std::env::var("PREDICAI_PORT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8080);
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", port)).await?;
    tracing::info!("predicai-edge listening on http://127.0.0.1:{port}");
    axum::serve(listener, router(state)).await?;
    Ok(())
}
