/* Types for src/data/platform_data.json, produced by
   dashboard/platform_export.py from platform/experiments/ (INDEX.json +
   record JSONs, read at export time) and platform/DATASET_REGISTRY.md
   (dataset facts transcribed in the export script). Keep in sync with
   that script. */

export type RecordKind = "experiment" | "transfer" | "explainability";

export interface PlatformDataset {
  id: string;
  name: string;
  source: string;
  licence: string;
  tasks: string[];
  status: "integrated" | "blocked" | "researched" | "not-found";
  adapter: string | null;
  note: string;
  /** Label/value spec pairs transcribed from the registry notes;
      empty for blocked / researched-only entries. */
  specs: [string, string][];
  records: number;
}

export interface ExperimentRecord {
  id: string;
  dataset: string;
  task: string;
  model: string;
  kind: RecordKind;
  seed: number | null;
  timestamp_utc: string | null;
  headline: Record<string, number | string>;
  train_s: number | null;
  infer_ms_per_row: number | null;
  protocol: string | null;
  notes: string | null;
}

export interface RefMetrics {
  label: string;
  rmse?: number;
  mae?: number;
  nasa_score?: number;
  accuracy?: number;
  f1_macro?: number;
}

export interface TransferRecord extends ExperimentRecord {
  source_dataset: string | null;
  target_dataset: string | null;
  cross_metrics: {
    rmse?: number;
    mae?: number;
    nasa_score?: number;
    pooled_rmse?: number;
    pooled_mae?: number;
    accuracy?: number;
    f1_macro?: number;
  };
  in_domain_references: RefMetrics[];
  degradation: Record<string, number | string> | null;
  verdict: string | null;
}

export interface XaiFeature {
  feature: string;
  value: number | null;
  value_label: string;
}

export interface XaiRecord extends ExperimentRecord {
  method: string;
  ranking: XaiFeature[];
  ranking_key: string;
}

export interface BestResult {
  dataset: string;
  task: string;
  experiment_id: string;
  model: string;
  metric_label: string;
  metric_value: number | null;
  metric_display: string | null;
  protocol: string | null;
}

export interface ImsBearingAlarm {
  bearing: string;
  documented_defect: string | null;
  first_sustained_alarm_index: number | null;
  first_sustained_alarm_timestamp: string | null;
  snapshots_of_warning_before_run_end: number;
  fraction_of_run_before_alarm: number | null;
  last_decile_exceedance_rate: number;
}

export interface PlatformData {
  generated_at: string;
  source: string;
  counts: {
    records: number;
    transfers: number;
    explainability: number;
    trained_model_records: number;
    datasets_integrated: number;
    datasets_registry_entries: number;
    tasks: Record<string, number>;
  };
  datasets: PlatformDataset[];
  experiments: ExperimentRecord[];
  transfers: TransferRecord[];
  xai: XaiRecord[];
  best_results: BestResult[];
  ims_alarms: {
    experiment_id: string;
    n_snapshots: number;
    framing: string;
    protocol: string;
    bearings: ImsBearingAlarm[];
  } | null;
}
