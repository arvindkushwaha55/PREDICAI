//! Axum routes for Phase D (plan.md §3):
//! - GET  /health                     — server up? model loaded? contract?
//! - POST /predict_rul                — validate -> normalize -> ONNX -> RUL (cycles)
//! - GET  /machines/{id}/history     — service history (SQLite, same DB as Python)
//!
//! Error policy: every failure is structured JSON with an HTTP status that
//! means something — 400 bad window, 503 model not loaded, 500 anything else.
//! A missing ONNX file is a 503, never a crash at startup: the server must
//! stay up so /health can report `model_loaded: false` until the evening
//! Kaggle run lands.

use crate::maintenance::Store;
use crate::model::RulModel;
use crate::preprocess::Preprocessing;
use crate::types::{ErrorResponse, HealthResponse, PredictRequest, PredictResponse};
use axum::{
    extract::{Path, State},
    http::StatusCode,
    routing::get,
    Json, Router,
};
use std::sync::{Arc, Mutex};

pub struct AppState {
    pub preprocessing: Preprocessing,
    /// None until `cmapss_lstm.onnx` exists (evening Kaggle run).
    pub model: Mutex<Option<RulModel>>,
    pub model_path: Option<String>,
    pub store: Store,
}

pub type SharedState = Arc<AppState>;

fn err(status: StatusCode, error: &str, detail: String) -> (StatusCode, Json<ErrorResponse>) {
    (status, Json(ErrorResponse { error: error.to_string(), detail }))
}

pub fn router(state: SharedState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/predict_rul", axum::routing::post(predict_rul))
        .route("/machines/{id}/history", get(machine_history))
        .with_state(state)
}

async fn health(State(state): State<SharedState>) -> Json<HealthResponse> {
    let loaded = state.model.lock().unwrap().is_some();
    Json(HealthResponse {
        status: "ok".to_string(),
        model_loaded: loaded,
        seq_len: state.preprocessing.seq_len,
        num_features: state.preprocessing.num_features,
        model_path: state.model_path.clone(),
    })
}

async fn predict_rul(
    State(state): State<SharedState>,
    Json(req): Json<PredictRequest>,
) -> Result<Json<PredictResponse>, (StatusCode, Json<ErrorResponse>)> {
    let p = &state.preprocessing;
    req.validate(p.seq_len, p.num_features)
        .map_err(|d| err(StatusCode::BAD_REQUEST, "invalid_window", d))?;

    let mut model = state.model.lock().unwrap();
    let Some(model) = model.as_mut() else {
        return Err(err(
            StatusCode::SERVICE_UNAVAILABLE,
            "model_not_loaded",
            "cmapss_lstm.onnx is not available yet — run the evening Kaggle training and place the ONNX file at ml/artifacts/cmapss_lstm.onnx, then restart the server".to_string(),
        ));
    };

    let normalized = p.normalize(&req.window);
    let rul = model
        .predict(&normalized, p.seq_len, p.num_features)
        .map_err(|e| err(StatusCode::INTERNAL_SERVER_ERROR, "inference_failed", e.to_string()))?;

    // Best-effort logging: a logging failure must not fail a good prediction.
    if let Err(e) = state.store.log_prediction(&req.machine_id, None, rul, "cmapss_lstm.onnx") {
        tracing::warn!("predictions_log write failed: {e}");
    }

    Ok(Json(PredictResponse {
        machine_id: req.machine_id,
        predicted_rul: rul,
        unit: "cycles".to_string(),
        status: "ok".to_string(),
    }))
}

async fn machine_history(
    State(state): State<SharedState>,
    Path(id): Path<String>,
) -> Result<Json<Vec<crate::maintenance::ServiceRecord>>, (StatusCode, Json<ErrorResponse>)> {
    state
        .store
        .service_history(&id)
        .map(Json)
        .map_err(|e| err(StatusCode::INTERNAL_SERVER_ERROR, "db_error", e.to_string()))
}
