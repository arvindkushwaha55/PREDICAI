import rawPlatform from "../data/platform_data.json";
import type { PlatformData } from "./platformTypes";

export const platform = rawPlatform as unknown as PlatformData;

/** Display name for a record/dataset id, e.g. "cmapss_fd001" -> "C-MAPSS FD001". */
export function datasetLabel(id: string): string {
  const known = platform.datasets.find((d) => d.id === id);
  if (known) return known.name;
  return id
    .replace(/^cmapss_/, "C-MAPSS ")
    .replace(/^ai4i2020$/, "AI4I 2020")
    .replace(/_/g, " ");
}

/** Short source->target label for transfer dataset ids like
    "cmapss_fd001_to_fd003" or "cwru_to_paderborn". */
export function transferPairLabel(
  source: string | null,
  target: string | null,
  fallbackId: string,
): string {
  if (source && target) return `${shortDs(source)} → ${shortDs(target)}`;
  return fallbackId;
}

function shortDs(id: string): string {
  return id
    .replace(/^cmapss_/, "")
    .replace(/^ai4i2020$/, "AI4I")
    .replace(/^nasa_battery$/, "Battery")
    .replace(/_/g, " ");
}

/** Format a metric value for display: thousands separators for big
    scores, more decimals for small ratios. */
export function fmtMetric(v: number): string {
  const a = Math.abs(v);
  if (a >= 1000)
    return v.toLocaleString("en-US", { maximumFractionDigits: 0 });
  if (a >= 100)
    return v.toLocaleString("en-US", {
      minimumFractionDigits: 1,
      maximumFractionDigits: 1,
    });
  if (a >= 1)
    return v.toLocaleString("en-US", {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    });
  return v.toLocaleString("en-US", {
    minimumFractionDigits: 3,
    maximumFractionDigits: 4,
  });
}

/** "test_rmse" -> "Test RMSE", "f1_macro" -> "F1 macro", etc. */
export function metricLabel(key: string): string {
  const map: Record<string, string> = {
    test_rmse: "Test RMSE",
    test_mae: "Test MAE",
    test_nasa: "Test NASA",
    pooled_rmse: "Pooled RMSE",
    pooled_mae: "Pooled MAE",
    f1: "F1",
    f1_macro: "F1-macro",
    roc_auc: "ROC-AUC",
    recall: "Recall",
    accuracy: "Accuracy",
    false_alarm_rate: "False-alarm rate",
    in_sample_rmse: "In-sample RMSE",
    top_feature: "Top feature",
    n_snapshots: "Snapshots",
  };
  return map[key] ?? key.replace(/_/g, " ");
}

export function fmtTrainTime(trainS: number | null): string {
  if (trainS == null) return "-";
  if (trainS === 0) return "0 s (no training, transfer record)";
  if (trainS < 60) return `${trainS.toFixed(1)} s`;
  return `${(trainS / 60).toFixed(1)} min`;
}
