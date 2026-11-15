"""PREDICAI platform — benchmarking waves 5-6 (cross-dataset
generalisation + explainability, 2026-11-15).

Wave 5 completes the transfer matrix that waves 2 and 4 opened, in
both directions, with the source-fitted preprocessing discipline of
the wave-2 transfer (the target dataset contributes NOTHING but its
test inputs and scoring labels):

- FD003 -> FD001 zero-shot transfer of the wave-4 CNN-LSTM (the saved
  checkpoint makes the reverse of wave 2's FD001 -> FD003 possible).
- FD002 <-> FD004 Ridge transfer in both directions (no LSTM
  checkpoints exist for these subsets; Ridge is deterministic, so the
  refit in this run IS the wave-2 in-domain model, restated in the
  record).
- CWRU <-> Paderborn bearing transfer in both directions over the
  shared label set {Normal, InnerRace, OuterRace} and the identical
  10 time-domain window features — across a 12 kHz vs 64 kHz sampling
  gap (a 2048-sample window spans 170.7 ms on CWRU, 32.0 ms on
  Paderborn), different rigs, bearings and loads.

Wave 6 adds the explainability records for the wave-4 models:

- SHAP (TreeExplainer) for a Battery Random Forest trained on all
  three EOL batteries (an explanation model — NOT the LOBO protocol;
  the difference is stated in the record), explained on B0005.
- Permutation importance for the Paderborn Random Forest on its
  wave-4 measurement-grouped test split.

And the IMS integration the user approved after wave 3's size gate:

- IMS Bearing Test 1 (NASA PCoE): Isolation Forest per bearing,
  fitted on the first 30% of snapshots (early-life nominal), scored
  over the full run. There are no per-snapshot labels, so the record
  reports ALARM TIMING against the teardown findings (bearing 3
  inner-race defect, bearing 4 roller-element defect), never
  classification scores.

Usage (from the platform/ directory):
    ../.venv/bin/python -m platform_lib.benchmarking.run_wave5 \
        --set fd003tofd001 | ridge | bearings | battery_shap |
              paderborn_perm | ims | all

Seed is 42 everywhere, matching waves 1-4.
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
from platform_lib.benchmarking.run_wave2 import _subset_data  # noqa: E402
from platform_lib.benchmarking.run_wave4 import (  # noqa: E402
    _battery_frame,
    _cwru_w4_frame,
    _reg_metrics,
)

ARTIFACTS_DIR = PLATFORM_DIR / "artifacts"
BEARING_LABELS = ["Normal", "InnerRace", "OuterRace"]


def _read_record(experiment_id: str) -> dict:
    return json.loads(
        (PLATFORM_DIR / "experiments" / f"{experiment_id}.json").read_text())


# --------------------------------------- 1) FD003 -> FD001 transfer
def run_transfer_fd003_to_fd001() -> None:
    import torch
    from torch import nn

    from predicai_ml.evaluate import evaluate_all
    from predicai_ml.model import predict_rul
    from predicai_ml.preprocessing import (Scaler, add_test_rul, clip_rul,
                                           last_window_per_engine)
    from platform_lib.adapters.cmapss import CMapssAdapter

    class CNNLSTMRegressor(nn.Module):
        """Identical copy of the wave-4 architecture (run_wave4.py) so
        the saved state dict loads unchanged."""

        def __init__(self, num_features, conv_1=64, conv_2=32,
                     hidden_1=128, hidden_2=64, head_dim=32, dropout=0.2):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv1d(num_features, conv_1, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(conv_1, conv_2, kernel_size=3, padding=1),
                nn.ReLU())
            self.lstm1 = nn.LSTM(conv_2, hidden_1, batch_first=True)
            self.drop1 = nn.Dropout(dropout)
            self.lstm2 = nn.LSTM(hidden_1, hidden_2, batch_first=True)
            self.drop2 = nn.Dropout(dropout)
            self.head = nn.Sequential(nn.Linear(hidden_2, head_dim),
                                      nn.ReLU(), nn.Linear(head_dim, 1))

        def forward(self, x):
            h = self.conv(x.transpose(1, 2)).transpose(1, 2)
            out, _ = self.lstm1(h)
            out = self.drop1(out)
            out, _ = self.lstm2(out)
            last = self.drop2(out[:, -1, :])
            return self.head(last).squeeze(-1)

    print("=== Transfer — FD003 CNN-LSTM -> FD001 (zero-shot) ===",
          flush=True)
    sidecar = json.loads(
        (ARTIFACTS_DIR / "fd003_cnn_lstm_preprocessing.json").read_text())
    features = sidecar["features"]
    arch = sidecar["architecture"]
    scaler = Scaler(features=list(features),
                    mean=np.array(sidecar["scaler"]["mean"],
                                  dtype=np.float64),
                    scale=np.array(sidecar["scaler"]["scale"],
                                   dtype=np.float64))

    adapter = CMapssAdapter(RAW / "cmapss" / "extracted", subset="FD001")
    test, rul = adapter.load_test()
    test = clip_rul(add_test_rul(test, rul), cap=sidecar["rul_cap"])
    test_scaled = scaler.transform(test)
    X_te, y_te, engines = last_window_per_engine(
        test_scaled, features, sidecar["seq_len"])
    print(f"  FD001 test last windows: {X_te.shape}; features "
          f"({len(features)}): {features}", flush=True)

    model = CNNLSTMRegressor(len(features), arch["conv_1"], arch["conv_2"],
                             arch["hidden_1"], arch["hidden_2"],
                             arch["head_dim"], arch["dropout"])
    state = torch.load(ARTIFACTS_DIR / "fd003_cnn_lstm.pt",
                       map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    x_te = torch.from_numpy(X_te)
    with torch.no_grad():
        pred_t, ms = timed_predict(lambda X: predict_rul(model, X), x_te,
                                   repeats=50)
    pred = pred_t.numpy()
    metrics = evaluate_all(y_te, pred)

    fd001_ref = json.loads((REPO_ROOT / "ml" / "artifacts" /
                            "metrics.json").read_text())[
        "test_last_window_per_engine"]
    fd003_cnn = _read_record("cmapss_fd003__taskB__cnn_lstm__s42")[
        "metrics"]["test_last_window_per_engine"]
    reverse = _read_record(
        "cmapss_fd001_to_fd003__taskB__transfer_lstm__s42")[
        "metrics"]["test_last_window_per_engine"]

    rec = base_record(
        "cmapss_fd003_to_fd001", "B", "transfer_cnn_lstm",
        {"architecture": "Conv1d(F->64,k3)->ReLU->Conv1d(64->32,k3)"
                         "->ReLU->LSTM(128)->LSTM(64)->head 32->1 "
                         "(wave-4 FD003 CNN-LSTM, weights unchanged)",
         "weights": "platform/artifacts/fd003_cnn_lstm.pt (wave-4 local "
                    "CPU run, best-val state)",
         "training": "none — zero-shot transfer, no FD001 fitting of "
                     "any kind (no fine-tuning, no scaler fitting)"})
    rec["experiment_id"] = ("cmapss_fd003_to_fd001__taskB__"
                            "transfer_cnn_lstm__s42")
    rec["split"] = {
        "protocol": "zero-shot cross-dataset transfer: wave-4 FD003 "
                    "CNN-LSTM applied to the official FD001 test set, "
                    "last window per engine; FD001 data used for "
                    "scoring only",
        "n_test_engines": int(len(X_te)),
        "source_dataset": "cmapss_fd003",
        "target_dataset": "cmapss_fd001",
    }
    rec["preprocessing"] = {
        "features": features,
        "n_features": len(features),
        "feature_set": "FD003's exact 15 features from the wave-4 "
                       "sidecar (FD003's own std-rule derivation — "
                       "all sensors except 1/5/10/16/18/19, settings "
                       "excluded; sensor_6 is KEPT, unlike FD001's 14)",
        "scaling": "z-score with FD003 train-split statistics from "
                   "platform/artifacts/fd003_cnn_lstm_preprocessing."
                   "json — source-fitted; no FD001 statistics used "
                   "anywhere",
        "rul_cap": sidecar["rul_cap"], "seq_len": sidecar["seq_len"],
    }
    rec["metrics"] = {"test_last_window_per_engine": metrics}
    rec["timing"] = {"train_s": 0.0, "infer_ms_per_row": round(ms, 5)}
    rec["notes"] = (
        "Zero-shot FD003 -> FD001 transfer, the mirror of wave 2's "
        "FD001 -> FD003 (RMSE 48.80). Asymmetry is expected and is "
        "itself the finding: the FD003 model was trained on TWO fault "
        "modes (HPC + fan) and a feature set that keeps sensor_6, "
        "while FD001 contains only the HPC mode — transferring INTO "
        "the narrower domain is not the same problem as transferring "
        "out of it. train_s = 0.0 because no training was performed "
        "in this run.")
    rec["extra"] = {
        "reference_fd003_in_domain_cnn_lstm": {
            "rmse": fd003_cnn["rmse"], "mae": fd003_cnn["mae"],
            "nasa_score": fd003_cnn["nasa_score"],
            "source": "platform record cmapss_fd003__taskB__cnn_lstm"
                      "__s42 (local CPU run, FD003-native 15 features)"},
        "reference_fd001_in_domain_shipped_lstm": {
            "rmse": fd001_ref["rmse"], "mae": fd001_ref["mae"],
            "nasa_score": fd001_ref["nasa_score"],
            "source": "ml/artifacts/metrics.json (shipped FD001 model "
                      "on FD001 test, same last-window protocol)"},
        "reverse_direction_wave2": {
            "rmse": reverse["rmse"], "mae": reverse["mae"],
            "nasa_score": reverse["nasa_score"],
            "source": "platform record cmapss_fd001_to_fd003__taskB__"
                      "transfer_lstm__s42 (shipped FD001 LSTM -> FD003,"
                      " FD001's 14 features)"},
        "degradation_vs_fd001_in_domain": {
            "rmse_delta": metrics["rmse"] - fd001_ref["rmse"],
            "rmse_pct_increase": 100.0 * (metrics["rmse"] /
                                          fd001_ref["rmse"] - 1.0),
            "mae_delta": metrics["mae"] - fd001_ref["mae"],
            "nasa_delta": metrics["nasa_score"] - fd001_ref["nasa_score"],
        },
        "per_engine": [
            {"engine": int(e), "y_true": float(t), "y_pred": float(p)}
            for e, t, p in zip(engines, y_te, pred)
        ],
    }
    save_record(rec, {"test_rmse": round(metrics["rmse"], 3),
                      "test_mae": round(metrics["mae"], 3),
                      "test_nasa": round(metrics["nasa_score"], 1)})


# ------------------------------------- 2) FD002 <-> FD004 Ridge xfer
def _ridge_transfer(src: str, dst: str) -> None:
    from sklearn.linear_model import Ridge

    from predicai_ml.evaluate import evaluate_all
    from predicai_ml.preprocessing import (add_test_rul, clip_rul,
                                           last_window_per_engine)
    from platform_lib.adapters.cmapss import CMapssAdapter

    print(f"=== Transfer — {src} Ridge -> {dst} (zero-shot) ===",
          flush=True)
    meta_s, data_s = _subset_data(src)
    meta_d, data_d = _subset_data(dst)
    same_features = data_s.features == data_d.features
    print(f"  source features ({len(data_s.features)}); target-derived "
          f"features identical: {same_features}", flush=True)
    assert same_features, (f"{src} and {dst} feature derivations differ: "
                           f"{data_s.features} vs {data_d.features}")

    Xtr = data_s.X_train.reshape(len(data_s.X_train), -1)
    model = Ridge(alpha=1.0)
    t0 = time.perf_counter()
    model.fit(Xtr, data_s.y_train)
    train_s = time.perf_counter() - t0

    adapter = CMapssAdapter(RAW / "cmapss" / "extracted", subset=dst)
    test, rul = adapter.load_test()
    test = clip_rul(add_test_rul(test, rul), cap=125)
    test_scaled = data_s.scaler.transform(test)
    X_te, y_te, engines = last_window_per_engine(
        test_scaled, data_s.features, 30)
    assert np.array_equal(y_te, data_d.y_test_last), (
        "transfer scoring labels disagree with the in-domain pipeline")
    X_te_flat = X_te.reshape(len(X_te), -1)
    pred, ms = timed_predict(
        lambda X: np.clip(model.predict(X), 0, None), X_te_flat,
        repeats=50)
    metrics = evaluate_all(y_te, pred)

    src_key, dst_key = f"cmapss_{src.lower()}", f"cmapss_{dst.lower()}"
    pair_key = f"cmapss_{src.lower()}_to_{dst.lower()}"
    dst_ridge = _read_record(f"{dst_key}__taskB__ridge_flat__s42")[
        "metrics"]["test_last_window_per_engine"]
    src_ridge = _read_record(f"{src_key}__taskB__ridge_flat__s42")[
        "metrics"]["test_last_window_per_engine"]

    rec = base_record(
        pair_key, "B", "transfer_ridge",
        {"alpha": 1.0,
         "input": "flattened (30 x F) windows, predictions clipped "
                  "at >= 0",
         "training": "REFIT in this run on the source subset with "
                     "the identical wave-2 config (Ridge alpha=1.0); "
                     "Ridge is deterministic, so this refit reproduces "
                     "the wave-2 in-domain model exactly — no wave-2 "
                     "checkpoint existed to load",
         "transfer": "the refit model + the SOURCE scaler are applied "
                     "to the target test set unchanged; the target "
                     "contributes only its test inputs and labels"})
    rec["experiment_id"] = f"{pair_key}__taskB__transfer_ridge__s42"
    rec["split"] = {
        "protocol": f"zero-shot cross-subset transfer: {src} Ridge "
                    f"(refit, identical config) applied to the "
                    f"official {dst} test set, last window per engine; "
                    f"{dst} data used for scoring only",
        "n_test_engines": int(len(X_te)),
        "n_train_windows_source": int(len(Xtr)),
        "source_dataset": src_key,
        "target_dataset": dst_key,
    }
    rec["preprocessing"] = {
        "features": data_s.features,
        "n_features": len(data_s.features),
        "feature_set": f"{src}'s derived features; {dst}'s own "
                       f"derivation produces the IDENTICAL list "
                       f"(verified in this run: {same_features}) — both "
                       f"are the 23 candidates that survive the "
                       f"std<0.005 rule on six-condition subsets (all 3 "
                       f"settings + 20 sensors)",
        "scaling": f"z-score with {src} train-split statistics "
                   f"(source-fitted; no {dst} statistics used)",
        "rul_cap": 125, "seq_len": 30,
    }
    rec["metrics"] = {"test_last_window_per_engine": metrics}
    rec["timing"] = {"train_s": round(train_s, 4),
                     "infer_ms_per_row": round(ms, 5)}
    rec["notes"] = (
        f"Zero-shot {src} -> {dst} Ridge transfer. Both subsets have "
        f"six operating conditions; {src} has "
        f"{meta_s['fault_modes']} fault mode(s), {dst} has "
        f"{meta_d['fault_modes']}. The shared 23-feature space makes "
        f"this the cleanest C-MAPSS transfer pair in the platform; "
        f"any degradation is therefore attributable to distribution / "
        f"fault-mode shift, not to feature misalignment. train_s is "
        f"the deterministic refit time on the source subset.")
    rec["extra"] = {
        "reference_target_in_domain_ridge": {
            "rmse": dst_ridge["rmse"], "mae": dst_ridge["mae"],
            "nasa_score": dst_ridge["nasa_score"],
            "source": f"platform record {dst_key}__taskB__ridge_flat"
                      f"__s42 (wave 2)"},
        "reference_source_in_domain_ridge": {
            "rmse": src_ridge["rmse"], "mae": src_ridge["mae"],
            "nasa_score": src_ridge["nasa_score"],
            "source": f"platform record {src_key}__taskB__ridge_flat"
                      f"__s42 (wave 2)"},
        "degradation_vs_target_in_domain": {
            "rmse_delta": metrics["rmse"] - dst_ridge["rmse"],
            "rmse_pct_increase": 100.0 * (metrics["rmse"] /
                                          dst_ridge["rmse"] - 1.0),
            "mae_delta": metrics["mae"] - dst_ridge["mae"],
            "nasa_delta": metrics["nasa_score"] - dst_ridge["nasa_score"],
        },
        "per_engine": [
            {"engine": int(e), "y_true": float(t), "y_pred": float(p)}
            for e, t, p in zip(engines, y_te, pred)
        ],
    }
    save_record(rec, {"test_rmse": round(metrics["rmse"], 3),
                      "test_mae": round(metrics["mae"], 3),
                      "test_nasa": round(metrics["nasa_score"], 1)})


def run_ridge_transfers() -> None:
    _ridge_transfer("FD002", "FD004")
    _ridge_transfer("FD004", "FD002")


# --------------------------------- 3) CWRU <-> Paderborn RF transfer
def _bearing_frames():
    from platform_lib.adapters.cwru import FEATURE_COLUMNS
    from platform_lib.adapters.paderborn import PaderbornAdapter

    cwru_df, catalogue = _cwru_w4_frame()
    pad_df, pad_meta = PaderbornAdapter(RAW / "paderborn").load()
    frames = {
        "cwru": (cwru_df[cwru_df["label"].isin(BEARING_LABELS)],
                 catalogue),
        "paderborn": (pad_df[pad_df["label"].isin(BEARING_LABELS)],
                      pad_meta),
    }
    return frames, FEATURE_COLUMNS


def _bearing_transfer(src: str, dst: str) -> None:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (accuracy_score, classification_report,
                                 confusion_matrix, f1_score)

    print(f"=== Transfer — {src} RF -> {dst} (bearings, zero-shot) ===",
          flush=True)
    frames, feature_cols = _bearing_frames()
    df_s, _ = frames[src]
    df_t, _ = frames[dst]
    X_s, y_s = df_s[feature_cols], df_s["label"]
    X_t, y_t = df_t[feature_cols], df_t["label"]
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
    in_domain = {
        "cwru": _read_record("cwru__taskA__random_forest_"
                             "recordingwise__s42"),
        "paderborn": _read_record("paderborn__taskA__random_forest__s42"),
    }
    rec = base_record(
        f"{src}_to_{dst}", "A", "transfer_rf",
        {"n_estimators": 300, "class_weight": "balanced", "n_jobs": -1,
         "features": "identical 10 per-window time-domain statistics "
                     "(adapters share window_features); window = 2048 "
                     "samples in BOTH datasets",
         "training": f"fit on ALL {src} windows of the shared label "
                     f"set; no target data in training"})
    rec["experiment_id"] = f"{src}_to_{dst}__taskA__transfer_rf__s42"
    rec["split"] = {
        "protocol": f"zero-shot cross-dataset transfer: Random Forest "
                    f"trained on all {src} windows, tested on all "
                    f"{dst} windows; label set restricted to the "
                    f"shared {BEARING_LABELS}",
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
        "features": feature_cols,
        "windowing": "non-overlapping 2048-sample windows in both "
                     "datasets — but the window TIME SPAN differs by "
                     "sampling rate: 2048 samples = 170.7 ms at CWRU's "
                     "12 kHz vs 32.0 ms at Paderborn's 64 kHz",
        "excluded": "CWRU 'Ball' class excluded (Paderborn scope has "
                    "no ball-fault bearing); CWRU windows of that class "
                    "are dropped from BOTH training (when CWRU is the "
                    "source) and scoring",
        "scaling": "none (tree model on raw time-domain features, as "
                   "in the in-domain records)",
    }
    rec["metrics"] = metrics
    rec["timing"] = {"train_s": round(train_s, 4),
                     "infer_ms_per_row": round(ms, 5)}
    rec["notes"] = (
        f"Zero-shot {src} -> {dst} bearing-fault transfer. Known shift "
        f"factors, all documented rather than corrected: sampling "
        f"rate 12 kHz (CWRU) vs 64 kHz (Paderborn) so identical "
        f"2048-sample windows cover 170.7 ms vs 32.0 ms of shaft "
        f"rotation; different test rigs, bearing geometries, sensor "
        f"mounting and load/speed regimes; CWRU faults are EDM-seeded "
        f"single-point defects, Paderborn's are EDM/drilling/engraving "
        f"defects on a different bearing type. Time-domain statistics "
        f"(RMS, kurtosis, ...) are amplitude- and rate-sensitive, so "
        f"degradation vs the in-domain references is the expected "
        f"result — this record measures it instead of assuming it.")
    rec["extra"] = {
        "reference_target_in_domain": {
            "accuracy": in_domain[dst]["metrics"]["accuracy"],
            "f1_macro": in_domain[dst]["metrics"]["f1_macro"],
            "protocol": in_domain[dst]["split"]["protocol"][:200],
            "source": f"platform record {in_domain[dst]['experiment_id']}"},
        "reference_source_in_domain": {
            "accuracy": in_domain[src]["metrics"]["accuracy"],
            "f1_macro": in_domain[src]["metrics"]["f1_macro"],
            "source": f"platform record {in_domain[src]['experiment_id']}"},
        "accuracy_delta_vs_target_in_domain":
            metrics["accuracy"] - in_domain[dst]["metrics"]["accuracy"],
    }
    save_record(rec, {"accuracy": round(metrics["accuracy"], 4),
                      "f1_macro": round(metrics["f1_macro"], 4)})


def run_bearing_transfers() -> None:
    _bearing_transfer("cwru", "paderborn")
    _bearing_transfer("paderborn", "cwru")


# --------------------------------------------- 5a) Battery SHAP (W6)
def run_battery_shap() -> None:
    import shap
    from sklearn.ensemble import RandomForestRegressor

    print("=== NASA Battery — Task B — SHAP (Random Forest) ===",
          flush=True)
    df, meta, features = _battery_frame()
    lobo = ["B0005", "B0006", "B0018"]
    d = df[df["battery"].isin(lobo)]
    X, y = d[features], d["rul_cycles_eol"]
    est = RandomForestRegressor(n_estimators=300, random_state=SEED,
                                n_jobs=-1)
    t0 = time.perf_counter()
    est.fit(X, y)
    train_s = time.perf_counter() - t0
    pred_all, ms = timed_predict(
        lambda A: np.clip(est.predict(A), 0, None), X)
    insample = _reg_metrics(y, pred_all)

    b5 = d[d["battery"] == "B0005"]
    t0 = time.perf_counter()
    explainer = shap.TreeExplainer(est)
    sv = explainer.shap_values(b5[features])
    shap_s = time.perf_counter() - t0
    sv = np.asarray(sv)
    if sv.ndim == 3:  # (n, features, outputs) in some shap versions
        sv = sv[:, :, 0]
    mean_abs = np.abs(sv).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    ranking = [{"feature": features[i],
                "mean_abs_shap": float(mean_abs[i])} for i in order]

    rec = base_record(
        "nasa_battery", "B", "shap_random_forest",
        {"model": "RandomForestRegressor(n_estimators=300) — the "
                  "wave-4 LOBO config, but refit on ALL cycles of "
                  "B0005/B0006/B0018 (explanation model, see split)",
         "explainer": "shap.TreeExplainer",
         "shap_version": shap.__version__,
         "explained_rows": "all discharge cycles of B0005",
         "n_explained_rows": int(len(b5))})
    rec["experiment_id"] = "battery__taskB__shap_random_forest__s42"
    rec["split"] = {
        "protocol": "EXPLANATION MODEL — deliberately NOT the wave-4 "
                    "leave-one-battery-out protocol: the forest is "
                    "fitted on every cycle of B0005, B0006 and B0018 so "
                    "the attributions describe the fullest model of "
                    "the three EOL batteries. Its metrics below are "
                    "IN-SAMPLE and must not be quoted as generalisation"
                    " — the honest generalisation estimate is the "
                    "wave-4 LOBO record (pooled RMSE 9.27).",
        "n_train_rows": int(len(d)),
        "batteries": lobo,
        "excluded_batteries": ["B0007 (censored — never crosses 1.4 Ah "
                               "EOL, no RUL label)"],
    }
    rec["preprocessing"] = {
        "features": features,
        "target": "rul_cycles_eol (discharge cycles until capacity "
                  "first < 1.4 Ah)",
        "note": "identical feature space to the wave-4 battery "
                "records (cycle_index + adapter FEATURE_COLUMNS)",
    }
    rec["metrics"] = {"in_sample_all_batteries": insample}
    rec["timing"] = {"train_s": round(train_s, 4),
                     "infer_ms_per_row": round(ms, 5)}
    rec["notes"] = (
        "Explainability record (wave 6): SHAP TreeExplainer "
        "attributions of the all-batteries Random Forest for B0005's "
        "discharge cycles, on the RUL (cycles) output scale. "
        "Attributions describe what THIS model uses — dominated, "
        "expectedly, by the capacity trajectory features — not a "
        "physical ageing mechanism. The LOBO record remains the "
        "performance reference for this dataset.")
    rec["extra"] = {
        "shap_mean_abs_ranking": ranking,
        "shap_compute_s": round(shap_s, 3),
        "lobo_reference": {
            "pooled_rmse": _read_record(
                "battery__taskB__random_forest_lobo__s42")[
                "metrics"]["pooled"]["rmse"],
            "source": "battery__taskB__random_forest_lobo__s42 "
                      "(wave 4, leave-one-battery-out)"},
    }
    save_record(rec, {"in_sample_rmse": round(insample["rmse"], 3),
                      "top_feature": ranking[0]["feature"]})


# --------------------------------- 5b) Paderborn permutation (W6)
def run_paderborn_permutation() -> None:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import GroupShuffleSplit

    from platform_lib.adapters.paderborn import PaderbornAdapter

    print("=== Paderborn — Task A — permutation importance (RF) ===",
          flush=True)
    df, meta = PaderbornAdapter(RAW / "paderborn").load()
    feature_cols = meta["feature_columns"]
    X, y = df[feature_cols], df["label"]
    groups = df["source_file"]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    tr_idx, te_idx = next(gss.split(X, y, groups))
    X_train, X_test = X.iloc[tr_idx], X.iloc[te_idx]
    y_train, y_test = y.iloc[tr_idx], y.iloc[te_idx]
    est = RandomForestClassifier(n_estimators=300,
                                 class_weight="balanced",
                                 random_state=SEED, n_jobs=-1)
    t0 = time.perf_counter()
    est.fit(X_train, y_train)
    train_s = time.perf_counter() - t0
    y_pred, ms = timed_predict(est.predict, X_test)
    acc = float(accuracy_score(y_test, y_pred))
    f1m = float(f1_score(y_test, y_pred, average="macro",
                         zero_division=0))
    print(f"  refit grouped-split accuracy {acc:.4f} (wave-4 record: "
          f"0.950)", flush=True)
    t0 = time.perf_counter()
    perm = permutation_importance(est, X_test, y_test, n_repeats=5,
                                  random_state=SEED, n_jobs=-1,
                                  scoring="accuracy")
    perm_s = time.perf_counter() - t0
    order = np.argsort(perm.importances_mean)[::-1]
    ranking = [{"feature": feature_cols[i],
                "importance_mean": float(perm.importances_mean[i]),
                "importance_std": float(perm.importances_std[i])}
               for i in order]

    rec = base_record(
        "paderborn", "A", "permutation_rf",
        {"model": "RandomForestClassifier(n_estimators=300, "
                  "class_weight='balanced') — identical config and "
                  "identical grouped split to the wave-4 record",
         "method": "sklearn permutation_importance on the grouped "
                   "test split, scoring='accuracy', n_repeats=5",
         "n_explained_rows": int(len(X_test))})
    rec["experiment_id"] = "paderborn__taskA__permutation_rf__s42"
    rec["split"] = {
        "protocol": "MEASUREMENT-GROUPED split identical to wave 4: "
                    "GroupShuffleSplit by source_file, 80/20, seed 42; "
                    "importance measured on the held-out measurements "
                    "only",
        "n_train": int(len(X_train)), "n_test": int(len(X_test)),
        "n_measurements_train": int(groups.iloc[tr_idx].nunique()),
        "n_measurements_test": int(groups.iloc[te_idx].nunique()),
    }
    rec["preprocessing"] = {
        "features": feature_cols,
        "windowing": "non-overlapping 2048-sample windows over the "
                     "vibration_1 channel (64 kHz), as wave 4",
        "bearings": meta["bearings"],
        "licence": meta["licence"],
    }
    rec["metrics"] = {"accuracy": acc, "f1_macro": f1m}
    rec["timing"] = {"train_s": round(train_s, 4),
                     "infer_ms_per_row": round(ms, 5)}
    rec["notes"] = (
        "Explainability record (wave 6): permutation importance of "
        "the wave-4 Paderborn forest, measured as accuracy drop on "
        "its grouped test split. The refit reproduces the wave-4 "
        "model (same config, split and seed), so the importances "
        "belong to that scored model. Caveat carried from wave 4: "
        "one physical bearing per class — importances describe "
        "within-scope separability, not unseen bearings.")
    rec["extra"] = {
        "permutation_importance_test_accuracy": ranking,
        "permutation_compute_s": round(perm_s, 3),
        "wave4_reference": {
            "accuracy": _read_record("paderborn__taskA__random_forest"
                                     "__s42")["metrics"]["accuracy"],
            "source": "paderborn__taskA__random_forest__s42 (wave 4)"},
    }
    save_record(rec, {"accuracy": round(acc, 4),
                      "top_feature": ranking[0]["feature"]})


# ------------------------------------------------------------------- CLI
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", required=True,
                        choices=["fd003tofd001", "ridge", "bearings",
                                 "battery_shap", "paderborn_perm", "ims",
                                 "all"])
    args = parser.parse_args()
    if args.set in ("fd003tofd001", "all"):
        run_transfer_fd003_to_fd001()
    if args.set in ("ridge", "all"):
        run_ridge_transfers()
    if args.set in ("bearings", "all"):
        run_bearing_transfers()
    if args.set in ("battery_shap", "all"):
        run_battery_shap()
    if args.set in ("paderborn_perm", "all"):
        run_paderborn_permutation()
    if args.set in ("ims", "all"):
        from platform_lib.benchmarking.run_wave5_ims import run_ims
        run_ims()


if __name__ == "__main__":
    main()
