/* Types for src/data/demo_data.json — produced by dashboard/data_export.py
   from real project artifacts (simulator/demo.db, ml/artifacts/*.json,
   edge parity fixtures). Keep in sync with that script. */

export interface SeriesPoint {
  step: number;
  end_cycle: number | null;
  predicted_rul: number;
  true_rul: number | null;
  created_at: string;
}

export interface EngineData {
  engine: number;
  machine_id: string;
  source: string;
  total_recorded_cycles: number;
  predictions_logged: number;
  cycle_alignment_verified: boolean;
  final: { predicted_rul: number; true_rul: number; abs_error: number };
  stats: {
    min_predicted_rul: number;
    max_predicted_rul: number;
    mean_predicted_rul: number;
    mean_abs_error_vs_true: number | null;
  };
  series: SeriesPoint[];
}

export interface ServiceRecord {
  record_id: number;
  machine_id: string;
  service_date: string;
  service_cycle: number;
  service_type: string;
  notes: string | null;
  is_synthetic: number;
  created_at: string;
}

export interface Recommendation {
  rec_id: number;
  machine_id: string;
  created_at: string;
  predicted_rul: number | null;
  current_cycle: number | null;
  cycles_since_service: number | null;
  level: string;
  reason: string;
  policy_version: string;
}

export interface MaintenanceData {
  machine_id: string;
  source: string;
  machine: {
    machine_id: string;
    display_name: string;
    registered_at: string;
    notes: string | null;
  } | null;
  service_records: ServiceRecord[];
  recommendations: Recommendation[];
  recommendation_inputs: {
    predicted_rul: number;
    predicted_rul_source: string;
    current_cycle: number;
    current_cycle_source: string;
  };
}

export interface MetricSet {
  rmse: number;
  mae: number;
  nasa_score: number;
  n: number;
}

export interface ModelData {
  source: string;
  architecture: string;
  model_file: string;
  weights: string;
  dataset: string;
  seq_len: number;
  num_features: number;
  features: string[];
  rul_cap: number;
  seed: number;
  onnx_opset: number;
  trained_on: string;
  training_config: Record<string, number>;
  metrics: {
    test_last_window_per_engine: MetricSet;
    validation_all_windows: MetricSet;
    best_val_rmse_during_training: number;
  };
  parity: {
    pytorch_vs_onnx_max_abs_diff: number;
    rust_vs_python_onnx: string;
  };
  explainability: {
    note: string;
    split: string;
    n_val_windows: number;
    permutation_top5: { sensor: string; rmse_increase: number }[];
    integrated_gradients_top5: {
      sensor: string;
      mean_abs_attribution: number;
      mean_signed_attribution: number;
    }[];
    top5_overlap: string[];
  };
}

export interface DemoData {
  generated_at: string;
  unit: string;
  policy_thresholds: {
    source: string;
    review_max_rul: number;
    inspection_max_rul: number;
    service_review_cycles: number;
  };
  engines: EngineData[];
  maintenance: MaintenanceData;
  model: ModelData;
  live_sample: {
    source: string;
    machine_id: string;
    window: number[][];
    expected_rul_from_parity_fixture: number;
    true_rul: number;
  };
}

/* Rust edge API payloads (edge/src/routes.rs). */
export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  seq_len: number;
  num_features: number;
  model_path?: string | null;
}

export interface PredictResponse {
  machine_id: string;
  predicted_rul: number;
  unit: string;
  status: string;
}
