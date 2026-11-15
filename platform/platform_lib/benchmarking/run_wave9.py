"""PREDICAI platform — benchmarking wave 9 (frequency-domain bearing
features + the honest protocols re-run, 2026-11-15).

Wave 9 attacks the two open problems the earlier waves documented
rather than solved:

1. Bearing fault LOCATION did not survive any honest protocol with
   the 10 time-domain window features: CWRU recording-wise
   (leave-one-recording-out) accuracy 0.665 / F1-macro 0.421 with
   InnerRace 0/177 correct (wave 4), and CWRU<->Paderborn zero-shot
   transfer 0.253 / 0.125 with Normal F1 = 0.0 in both directions
   (wave 5). Bearing faults are spectral phenomena, so this wave
   adds the 14 rate-free spectral/envelope features of
   platform_lib.features.spectral (24 combined features) and
   re-runs BOTH protocols with everything else held identical —
   same recordings, same splits, same Random Forest config, same
   seed. Whatever the outcome, it is reported as measured: if the
   spectral features do not fix the problem, that is the result.

2. IMS completeness (Tests 2-3) lives in run_wave9_ims.py.

Usage (from the platform/ directory):
    ../.venv/bin/python -m platform_lib.benchmarking.run_wave9 \
        --set cwru | transfers | all

Seed is 42 everywhere, matching waves 1-8.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PLATFORM_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = PLATFORM_DIR.parent
ML_SRC = REPO_ROOT / "ml" / "src"
if str(ML_SRC) not in sys.path:
    sys.path.insert(0, str(ML_SRC))
if str(PLATFORM_DIR) not in sys.path:
    sys.path.insert(0, str(PLATFORM_DIR))

from platform_lib.benchmarking.run_benchmark import (  # noqa: E402
    RAW,
    SEED,
    base_record,
    save_record,
    timed_predict,
)
from platform_lib.benchmarking.run_wave4 import (  # noqa: E402
    CWRU_W4_RECORDINGS,
)
from platform_lib.features.spectral import (  # noqa: E402
    COMBINED_FEATURE_COLUMNS,
    combined_window_features,
)

BEARING_LABELS = ["Normal", "InnerRace", "OuterRace"]

FEATURE_PROTOCOL_NOTE = (
    "combined wave-9 feature set: the 10 time-domain statistics of "
    "the earlier waves (untouched) + 14 spectral/envelope features "
    "(platform_lib.features.spectral). All spectral quantities are "
    "RATE-FREE: frequencies and band edges are fractions of Nyquist "
    "(8 log-spaced bands over 0.005..1.0 x Nyquist), band energies "
    "are normalised by total spectral energy, and the envelope peak "
    "amplitude is normalised by total envelope-spectrum magnitude — "
    "so the identical feature definitions apply at CWRU's 12 kHz "
    "and Paderborn's 64 kHz. This removes the sampling-rate confound "
    "only; it does not align shaft-speed-relative defect frequencies "
    "across rigs.")


def _read_record(experiment_id: str) -> dict:
    return json.loads(
        (PLATFORM_DIR / "experiments" / f"{experiment_id}.json").read_text())


# ------------------------------------------------------- spectral frames
def _cwru_spectral_frame() -> tuple[pd.DataFrame, list[dict]]:
    """The wave-4 CWRU recording catalogue, featurised with the
    combined set. Window extraction is identical to
    run_wave4._cwru_w4_frame (same files, same X<id> variable
    selection, same non-overlapping 2048-sample windows) — only the
    per-window feature function differs."""
    from scipy.io import loadmat

    from platform_lib.adapters.cwru import WINDOW_LEN

    rows, catalogue = [], []
    for fname, (condition, label) in CWRU_W4_RECORDINGS.items():
        path = RAW / "cwru" / fname
        if not path.exists():
            raise FileNotFoundError(f"CWRU recording missing: {path}")
        mat = loadmat(path)
        key = f"X{int(fname.split('.')[0]):03d}_DE_time"
        if key not in mat:
            raise ValueError(f"{fname}: expected variable {key}, found "
                             f"{[k for k in mat if k.endswith('_DE_time')]}")
        series = np.asarray(mat[key], dtype=np.float64).ravel()
        n_windows = len(series) // WINDOW_LEN
        for i in range(n_windows):
            feats = combined_window_features(
                series[i * WINDOW_LEN:(i + 1) * WINDOW_LEN])
            feats["label"] = label
            feats["recording"] = condition
            feats["source_file"] = fname
            rows.append(feats)
        catalogue.append({"file": fname, "condition": condition,
                          "label": label,
                          "n_samples": int(len(series)),
                          "n_windows": int(n_windows),
                          "official_url": "https://engineering.case.edu/"
                                          f"sites/default/files/{fname}"})
    return pd.DataFrame(rows), catalogue


def _paderborn_spectral_frame() -> tuple[pd.DataFrame, dict]:
    """All Paderborn windows with the combined feature set. Mirrors
    PaderbornAdapter.load() file-for-file (same bearings, same
    vibration_1 channel, same 2048-sample windows) — only the
    per-window feature function differs."""
    from platform_lib.adapters.paderborn import (BEARINGS,
                                                 load_vibration,
                                                 parse_condition)
    from platform_lib.adapters.cwru import WINDOW_LEN

    rows, per_bearing = [], []
    raw_dir = RAW / "paderborn"
    for code, (label, origin) in BEARINGS.items():
        files = sorted((raw_dir / code).glob("*.mat"))
        if not files:
            raise FileNotFoundError(f"no .mat files under {raw_dir / code}")
        n_windows = 0
        for path in files:
            series = load_vibration(path)
            cond = parse_condition(path.name.split(f"_{code}_")[0])
            for i in range(len(series) // WINDOW_LEN):
                feats = combined_window_features(
                    series[i * WINDOW_LEN:(i + 1) * WINDOW_LEN])
                feats.update(cond)
                feats["label"] = label
                feats["bearing"] = code
                feats["damage_origin"] = origin
                feats["source_file"] = path.name
                rows.append(feats)
                n_windows += 1
        per_bearing.append({"bearing": code, "label": label,
                            "damage_origin": origin,
                            "n_measurements": int(len(files)),
                            "n_windows": int(n_windows)})
    meta = {"bearings": per_bearing, "n_windows": int(len(rows))}
    return pd.DataFrame(rows), meta


# ------------------------------------- 1) CWRU recording-wise, spectral
def run_cwru_recordingwise_spectral() -> None:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (accuracy_score, classification_report,
                                 confusion_matrix, f1_score)

    from platform_lib.adapters.cwru import LABELS

    print("=== CWRU — Task A — RF, RECORDING-WISE (LOGO), combined "
          "spectral features ===", flush=True)
    df, catalogue = _cwru_spectral_frame()
    print(f"  windows: {len(df)} over {df['recording'].nunique()} "
          f"recordings; per class: {df['label'].value_counts().to_dict()}",
          flush=True)
    X = df[COMBINED_FEATURE_COLUMNS]
    y = df["label"]
    groups = df["recording"]

    y_pred = pd.Series(index=df.index, dtype=object)
    per_recording, total_train_s, total_pred_s = [], 0.0, 0.0
    for rec_id in sorted(groups.unique()):
        te_mask = groups == rec_id
        est = RandomForestClassifier(n_estimators=300,
                                     class_weight="balanced",
                                     random_state=SEED, n_jobs=-1)
        t0 = time.perf_counter()
        est.fit(X[~te_mask], y[~te_mask])
        total_train_s += time.perf_counter() - t0
        t0 = time.perf_counter()
        pred = est.predict(X[te_mask])
        total_pred_s += time.perf_counter() - t0
        y_pred[te_mask] = pred
        per_recording.append({
            "recording": rec_id, "label": str(y[te_mask].iloc[0]),
            "n_windows": int(te_mask.sum()),
            "accuracy": float(accuracy_score(y[te_mask], pred))})
        print(f"    held out {rec_id:12s} "
              f"acc {per_recording[-1]['accuracy']:.3f}", flush=True)

    labels = [c for c in LABELS if c in set(y)]
    report = classification_report(y, y_pred, labels=labels,
                                   output_dict=True, zero_division=0)
    cm = confusion_matrix(y, y_pred, labels=labels).tolist()
    metrics = {
        "accuracy": float(accuracy_score(y, y_pred)),
        "f1_macro": float(f1_score(y, y_pred, average="macro",
                                   zero_division=0)),
        "confusion_matrix": cm,
        "confusion_labels": labels,
        "per_class": {c: {"precision": report[c]["precision"],
                          "recall": report[c]["recall"],
                          "f1": report[c]["f1-score"],
                          "support": int(report[c]["support"])}
                      for c in labels},
    }
    wave4 = _read_record("cwru__taskA__random_forest_recordingwise__s42")
    ir_idx = labels.index("InnerRace")
    comparison = {
        "wave4_time_domain_only": {
            "accuracy": wave4["metrics"]["accuracy"],
            "f1_macro": wave4["metrics"]["f1_macro"],
            "innerrace_correct": wave4["metrics"]["confusion_matrix"][ir_idx][ir_idx],
            "innerrace_support": wave4["metrics"]["per_class"]["InnerRace"]["support"],
            "innerrace_f1": wave4["metrics"]["per_class"]["InnerRace"]["f1"],
            "record": "cwru__taskA__random_forest_recordingwise__s42"},
        "wave9_combined_spectral": {
            "accuracy": metrics["accuracy"],
            "f1_macro": metrics["f1_macro"],
            "innerrace_correct": cm[ir_idx][ir_idx],
            "innerrace_support": metrics["per_class"]["InnerRace"]["support"],
            "innerrace_f1": metrics["per_class"]["InnerRace"]["f1"]},
        "delta_accuracy": metrics["accuracy"] - wave4["metrics"]["accuracy"],
        "delta_f1_macro": metrics["f1_macro"] - wave4["metrics"]["f1_macro"],
    }
    rec = base_record(
        "cwru", "A", "random_forest_spectral_recordingwise",
        {"n_estimators": 300, "class_weight": "balanced", "n_jobs": -1,
         "features": FEATURE_PROTOCOL_NOTE})
    rec["split"] = {
        "protocol": "RECORDING-WISE leave-one-recording-out "
                    "(LeaveOneGroupOut over the SAME 12 recordings as "
                    "wave 4; groups = recording id): in every fold the "
                    "held-out recording's windows are the test set and "
                    "NO window from that recording appears in "
                    "training. Metrics are computed on the pooled "
                    "out-of-fold predictions. Protocol, recordings, "
                    "model config and seed are identical to wave 4 — "
                    "the ONLY change is the feature set.",
        "n_recordings": int(df["recording"].nunique()),
        "n_windows": int(len(df)),
        "recordings_per_class": {k: int(v) for k, v in
                                 y.groupby(y).count().items()},
        "class_window_counts": {k: int(v) for k, v in
                                y.value_counts().items()},
    }
    rec["preprocessing"] = {
        "features": COMBINED_FEATURE_COLUMNS,
        "feature_note": FEATURE_PROTOCOL_NOTE,
        "windowing": "non-overlapping windows of 2048 samples over "
                     "the DE channel of each recording (identical "
                     "to waves 2/4)",
        "recordings": catalogue,
    }
    rec["timing"] = {"train_s": round(total_train_s, 4),
                     "infer_ms_per_row": round(
                         total_pred_s / len(df) * 1000.0, 5)}
    rec["metrics"] = metrics
    rec["notes"] = (
        "Wave-9 counterpart of the wave-4 recording-wise record: "
        "identical protocol, recordings, model and seed; the only "
        "change is the feature set (10 time-domain -> 24 combined "
        "time + spectral/envelope). The direct comparison against "
        "wave 4 lives in extra.comparison_wave4_time_domain. "
        "Recording-wise evaluation still does NOT cover unseen fault "
        "diameters combined with unseen loads jointly, nor different "
        "machines. Data note (as wave 4): 99.mat as served bundles "
        "X098 alongside X099; the self-identified X099_DE_time "
        "series (2 hp normal) is the one used.")
    rec["extra"] = {
        "per_recording_accuracy": per_recording,
        "comparison_wave4_time_domain": comparison,
    }
    save_record(rec, {"accuracy": round(metrics["accuracy"], 4),
                      "f1_macro": round(metrics["f1_macro"], 4),
                      "innerrace_correct": cm[ir_idx][ir_idx]})


# --------------------------------- 2) CWRU <-> Paderborn transfer, spectral
def _bearing_frames_spectral():
    cwru_df, catalogue = _cwru_spectral_frame()
    pad_df, pad_meta = _paderborn_spectral_frame()
    frames = {
        "cwru": (cwru_df[cwru_df["label"].isin(BEARING_LABELS)],
                 catalogue),
        "paderborn": (pad_df[pad_df["label"].isin(BEARING_LABELS)],
                      pad_meta),
    }
    return frames


def _bearing_transfer_spectral(src: str, dst: str) -> None:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (accuracy_score, classification_report,
                                 confusion_matrix, f1_score)

    print(f"=== Transfer — {src} RF -> {dst} (bearings, zero-shot, "
          f"combined spectral features) ===", flush=True)
    frames = _bearing_frames_spectral()
    df_s, _ = frames[src]
    df_t, _ = frames[dst]
    X_s, y_s = df_s[COMBINED_FEATURE_COLUMNS], df_s["label"]
    X_t, y_t = df_t[COMBINED_FEATURE_COLUMNS], df_t["label"]
    print(f"  source {src}: {len(df_s)} windows "
          f"{y_s.value_counts().to_dict()}; target {dst}: {len(df_t)} "
          f"windows {y_t.value_counts().to_dict()}", flush=True)

    est = RandomForestClassifier(n_estimators=300,
                                 class_weight="balanced",
                                 random_state=SEED, n_jobs=-1)
    t0 = time.perf_counter()
    est.fit(X_s, y_s)
    train_s = time.perf_counter() - t0
    y_pred, ms = timed_predict(est.predict, X_t)
    report = classification_report(y_t, y_pred, labels=BEARING_LABELS,
                                   output_dict=True, zero_division=0)
    metrics = {
        "accuracy": float(accuracy_score(y_t, y_pred)),
        "f1_macro": float(f1_score(y_t, y_pred, average="macro",
                                   zero_division=0)),
        "confusion_matrix": confusion_matrix(
            y_t, y_pred, labels=BEARING_LABELS).tolist(),
        "confusion_labels": BEARING_LABELS,
        "per_class": {c: {"precision": report[c]["precision"],
                          "recall": report[c]["recall"],
                          "f1": report[c]["f1-score"],
                          "support": int(report[c]["support"])}
                      for c in BEARING_LABELS},
    }
    wave5 = _read_record(f"{src}_to_{dst}__taskA__transfer_rf__s42")
    in_domain = {
        "cwru": _read_record("cwru__taskA__random_forest_"
                             "recordingwise__s42"),
        "paderborn": _read_record("paderborn__taskA__random_forest__s42"),
    }
    rec = base_record(
        f"{src}_to_{dst}", "A", "transfer_rf_spectral",
        {"n_estimators": 300, "class_weight": "balanced", "n_jobs": -1,
         "features": FEATURE_PROTOCOL_NOTE,
         "training": f"fit on ALL {src} windows of the shared label "
                     f"set; no target data in training"})
    rec["experiment_id"] = f"{src}_to_{dst}__taskA__transfer_rf_spectral__s42"
    rec["split"] = {
        "protocol": f"zero-shot cross-dataset transfer: Random Forest "
                    f"trained on all {src} windows, tested on all "
                    f"{dst} windows; label set restricted to the "
                    f"shared {BEARING_LABELS}. Identical to the "
                    f"wave-5 transfer protocol — the ONLY change is "
                    f"the feature set (10 time-domain -> 24 combined).",
        "source_dataset": src,
        "target_dataset": dst,
        "n_train_windows": int(len(df_s)),
        "n_test_windows": int(len(df_t)),
        "train_class_counts": {k: int(v) for k, v in
                               y_s.value_counts().items()},
        "test_class_counts": {k: int(v) for k, v in
                              y_t.value_counts().items()},
    }
    rec["preprocessing"] = {
        "features": COMBINED_FEATURE_COLUMNS,
        "feature_note": FEATURE_PROTOCOL_NOTE,
        "windowing": "non-overlapping 2048-sample windows in both "
                     "datasets — the window TIME SPAN still differs "
                     "by sampling rate: 2048 samples = 170.7 ms at "
                     "CWRU's 12 kHz vs 32.0 ms at Paderborn's 64 kHz. "
                     "The spectral features are computed per window "
                     "in rate-free units, but a window on one rig "
                     "still observes ~5.3x more shaft revolutions "
                     "than on the other.",
        "excluded": "CWRU 'Ball' class excluded (Paderborn scope has "
                    "no ball-fault bearing); CWRU windows of that "
                    "class are dropped from BOTH training (when CWRU "
                    "is the source) and scoring",
        "scaling": "none (tree model, as in the wave-5 records)",
    }
    rec["metrics"] = metrics
    rec["timing"] = {"train_s": round(train_s, 4),
                     "infer_ms_per_row": round(ms, 5)}
    rec["notes"] = (
        f"Zero-shot {src} -> {dst} bearing-fault transfer with the "
        f"wave-9 combined spectral feature set — the wave-5 transfer "
        f"re-run with the feature set as the only changed variable. "
        f"Shift factors that remain, documented rather than "
        f"corrected: different test rigs, bearing geometries, sensor "
        f"mounting and load/speed regimes; shaft-speed-relative "
        f"defect frequencies are NOT aligned by the rate-free "
        f"features (sampling rate and shaft speed vary "
        f"independently); window time spans differ 5.3x. The wave-5 "
        f"time-domain baseline for this direction is carried in "
        f"extra.wave5_time_domain_baseline; the result is reported "
        f"as measured, whether or not it improves.")
    rec["extra"] = {
        "wave5_time_domain_baseline": {
            "accuracy": wave5["metrics"]["accuracy"],
            "f1_macro": wave5["metrics"]["f1_macro"],
            "normal_precision": wave5["metrics"]["per_class"]["Normal"]["precision"],
            "normal_recall": wave5["metrics"]["per_class"]["Normal"]["recall"],
            "normal_f1": wave5["metrics"]["per_class"]["Normal"]["f1"],
            "record": f"{src}_to_{dst}__taskA__transfer_rf__s42"},
        "delta_vs_wave5_time_domain": {
            "accuracy_delta": metrics["accuracy"]
                              - wave5["metrics"]["accuracy"],
            "f1_macro_delta": metrics["f1_macro"]
                              - wave5["metrics"]["f1_macro"],
            "normal_f1_delta": metrics["per_class"]["Normal"]["f1"]
                               - wave5["metrics"]["per_class"]["Normal"]["f1"]},
        "reference_target_in_domain": {
            "accuracy": in_domain[dst]["metrics"]["accuracy"],
            "f1_macro": in_domain[dst]["metrics"]["f1_macro"],
            "protocol": in_domain[dst]["split"]["protocol"][:200],
            "source": f"platform record {in_domain[dst]['experiment_id']} "
                      f"(time-domain features)"},
        "reference_source_in_domain": {
            "accuracy": in_domain[src]["metrics"]["accuracy"],
            "f1_macro": in_domain[src]["metrics"]["f1_macro"],
            "source": f"platform record {in_domain[src]['experiment_id']} "
                      f"(time-domain features)"},
        "accuracy_delta_vs_target_in_domain":
            metrics["accuracy"] - in_domain[dst]["metrics"]["accuracy"],
    }
    save_record(rec, {"accuracy": round(metrics["accuracy"], 4),
                      "f1_macro": round(metrics["f1_macro"], 4),
                      "normal_f1": round(
                          metrics["per_class"]["Normal"]["f1"], 4)})


def run_bearing_transfers_spectral() -> None:
    _bearing_transfer_spectral("cwru", "paderborn")
    _bearing_transfer_spectral("paderborn", "cwru")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", default="all",
                        choices=["cwru", "transfers", "all"])
    args = parser.parse_args()
    if args.set in ("cwru", "all"):
        run_cwru_recordingwise_spectral()
    if args.set in ("transfers", "all"):
        run_bearing_transfers_spectral()
