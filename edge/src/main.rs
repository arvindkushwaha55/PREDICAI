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

    /* Exported dashboard data, served at /data/demo and /data/platform
       when the files exist (the hosted deployment bakes them in). A
       missing file is not fatal: those routes answer 404 and the rest
       of the API works as before. */
    let load_data = |env: &str, fallback: &str| -> Option<String> {
        let path = default_path(env, fallback);
        match std::fs::read_to_string(&path) {
            Ok(text) => {
                tracing::info!("dashboard data loaded: {}", path.display());
                Some(text)
            }
            Err(_) => {
                tracing::warn!("dashboard data not found (route will 404): {}", path.display());
                None
            }
        }
    };
    let demo_data = load_data("PREDICAI_DEMO_DATA", "../dashboard/src/data/demo_data.json");
    let platform_data = load_data(
        "PREDICAI_PLATFORM_DATA",
        "../dashboard/src/data/platform_data.json",
    );

    /* /health reports only the model file NAME, never a server path. */
    let model_name = model_path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned());

    let state = Arc::new(AppState {
        preprocessing,
        model: Mutex::new(model),
        model_path: model_name,
        store,
        demo_data,
        platform_data,
    });

    /* Hosted platforms inject PORT; PREDICAI_PORT stays the local
       override. Binding stays on 127.0.0.1 unless PREDICAI_BIND says
       otherwise (the hosted deployment sets 0.0.0.0). */
    let port: u16 = std::env::var("PORT")
        .ok()
        .and_then(|v| v.parse().ok())
        .or_else(|| {
            std::env::var("PREDICAI_PORT")
                .ok()
                .and_then(|v| v.parse().ok())
        })
        .unwrap_or(8080);
    let bind = std::env::var("PREDICAI_BIND").unwrap_or_else(|_| "127.0.0.1".to_string());
    let listener = tokio::net::TcpListener::bind((bind.as_str(), port)).await?;
    tracing::info!("predicai-edge listening on http://{bind}:{port}");
    axum::serve(listener, router(state)).await?;
    Ok(())
}
