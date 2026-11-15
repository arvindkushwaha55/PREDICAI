"""PREDICAI platform — benchmarking wave 12 (2026-11-15).

Wave 12 runs the two principled items the earlier waves documented
but had not executed:

1. FD004 PER-REGIME LSTM. Wave 10 re-ran FD002/FD004 Ridge and the
   FD002 LSTM with per-regime normalisation (FD002 LSTM 16.236 ->
   14.046) and deliberately skipped the FD004 LSTM for CPU budget.
   This is that run: identical protocol to the wave-10 FD002
   per-regime LSTM (same regime pipeline, same architecture and
   hyperparameters, cap 125, window 30, engine-wise split seed 42);
   only the subset changes. The record carries BOTH comparisons:
   the wave-2 global-normalisation FD004 LSTM (18.809) and the
   wave-10 FD004 per-regime Ridge (19.115).

2. ORDER-DOMAIN BEARING FEATURES. Waves 4/5/9 showed fault
   *location* does not transfer across rigs with time-domain
   features (transfers 0.253/0.125) or with wave 9's rate-free
   spectral features (0.210/0.187), whose own design note states
   they "do not align shaft-speed-relative defect frequencies".
   Wave 12 adds the 9 order-domain features of
   platform_lib.features.order (envelope-spectrum energy shares at
   BPFO/BPFI/BSF/FTF and their 2nd harmonics, per-dataset bearing
   geometry, per-recording shaft speeds) to the untouched 10
   time-domain features and re-runs the wave-9 protocols EXACTLY —
   same recordings, same splits, same Random Forest config, same
   seed: CWRU recording-wise LOGO and both CWRU<->Paderborn
   zero-shot transfers. Whatever the outcome, it is reported as
   measured; if order alignment also fails, the cross-rig transfer
   question is answered for this feature family and the records
   say so.

Usage (from the platform/ directory):
    ../.venv/bin/python -m platform_lib.benchmarking.run_wave12 \
        --set fd004-lstm | cwru | transfers | all

Seed is 42 everywhere, matching waves 1-11.
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
from platform_lib.benchmarking.run_wave2 import LSTM_CFG  # noqa: E402
from platform_lib.benchmarking.run_wave4 import (  # noqa: E402
    CWRU_W4_RECORDINGS,
)
from platform_lib.benchmarking.run_wave10 import (  # noqa: E402
    _perregime_record,
    _subset_data_perregime,
    _wave2_test_metrics,
)
from platform_lib.features.order import (  # noqa: E402
    CWRU_6205,
    ORDER_COMBINED_FEATURE_COLUMNS,
    PADERBORN_6203,
    order_window_features,
)

BEARING_LABELS = ["Normal", "InnerRace", "OuterRace"]

FEATURE_PROTOCOL_NOTE = (
    "wave-12 feature set: the 10 time-domain statistics of the "
    "earlier waves (untouched) + 9 order-domain features "
    "(platform_lib.features.order): envelope-spectrum energy "
    "shares in +-2% bands at the bearing's BPFO, BPFI, BSF "
    "(ball-defect order = 2*ball-spin) and FTF characteristic "
    "orders and their 2nd harmonics, plus order-domain spectral "
    "entropy. Orders are computed per dataset from the actual "
    "bearing geometry (CWRU: SKF 6205-2RS JEM, 9 balls, Bd 0.3126 "
    "in, Pd 1.537 in -> BPFO 3.5848 / BPFI 5.4152 / BSF 4.7134 / "
    "FTF 0.3983; Paderborn: 6203, 8 balls, Bd 6.75 mm, Pd 28.55 mm "
    "-> BPFO 3.0543 / BPFI 4.9457 / BSF 3.9932 / FTF 0.3818) and "
    "each window is read at ITS recording's shaft speed, so the "
    "features are shaft-speed-relative by construction. Residual "
    "confounds, stated not hidden: true order resolution is "
    "1/(revolutions per window) — 0.196 orders on CWRU windows "
    "(5.11 rev) but 1.25/2.08 orders on Paderborn windows "
    "(0.80 rev at N15, 0.48 rev at N09), coarser than the spacing "
    "between characteristic orders; slip moves true defect rates "
    "~1-2% off the kinematic orders (hence the +-2% bands); rig "
    "transfer functions and sensor placement still differ.")


def _read_record(experiment_id: str) -> dict:
    return json.loads(
        (PLATFORM_DIR / "experiments" / f"{experiment_id}.json").read_text())


# ------------------------------------------------- 1) FD004 per-regime LSTM
def run_fd004_lstm_perregime() -> None:
    """Verbatim wave-10 FD002 per-regime LSTM protocol, subset FD004.

    The training loop below is run_wave10.run_subset_lstm_perregime
    unchanged (same LSTM_CFG, same early stopping, same restore);
    only the subset and the finish (two comparison blocks) differ.
    """
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    from predicai_ml.evaluate import evaluate_all
    from predicai_ml.model import LSTMRULRegressor, predict_rul

    subset = "FD004"
    print(f"=== C-MAPSS {subset} — Task B — LSTM, PER-REGIME scaling "
          f"(wave 12) ===", flush=True)
    meta, data = _subset_data_perregime(subset)
    ri = data.regime_info
    print(f"  features ({len(data.features)}); regime train sizes: "
          f"{ri['train_sizes']}", flush=True)
    print(f"  train windows {len(data.X_train)} / val {len(data.X_val)}",
          flush=True)
    cfg = dict(LSTM_CFG)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    rec = _perregime_record(
        subset, "lstm_perregime",
        {**cfg, "device": "cpu (local run)",
         "architecture": "LSTM(128)->LSTM(64)->head 32->1 (same as "
                         "wave 2 / shipped FD001 recipe)"},
        meta, data, wave2_model="lstm")
    rec["notes"] = rec["notes"].replace(
        "Wave-10 per-regime normalisation re-run of the wave-2 lstm "
        "experiment",
        "Wave-12 execution of the wave-10 per-regime protocol for "
        "the FD004 LSTM — the item wave 10 deliberately skipped for "
        "CPU budget; protocol, features, cap, windowing and model "
        "are the wave-10 per-regime ones, unchanged")

    ds = TensorDataset(torch.from_numpy(data.X_train),
                       torch.from_numpy(data.y_train))
    dl = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True)
    x_va = torch.from_numpy(data.X_val)
    x_te = torch.from_numpy(data.X_test_last)
    y_va = torch.from_numpy(data.y_val)

    model = LSTMRULRegressor(len(data.features), cfg["hidden_1"],
                             cfg["hidden_2"], cfg["head_dim"],
                             cfg["dropout"])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                            weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg["epochs"])
    loss_fn = nn.MSELoss()

    best_rmse, best_state, bad_epochs, epochs_run = float("inf"), None, 0, 0
    t0 = time.time()
    for epoch in range(1, cfg["epochs"] + 1):
        epochs_run = epoch
        model.train()
        for xb, yb in dl:
            opt.zero_grad()
            loss = loss_fn(model(xb).squeeze(-1), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step()
        sched.step()
        with torch.no_grad():
            val_rmse = float(torch.sqrt(torch.mean(
                (predict_rul(model, x_va) - y_va) ** 2)))
        print(f"  fd004-lstm-perregime epoch {epoch:02d}  "
              f"val RMSE {val_rmse:.3f}  ({(time.time() - t0) / 60:.1f} min)",
              flush=True)
        if val_rmse < best_rmse - 1e-4:
            best_rmse, bad_epochs = val_rmse, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= cfg["patience"]:
                print(f"  early stop at epoch {epoch}", flush=True)
                break
    train_s = time.time() - t0
    model.load_state_dict(best_state)
    rec["timing"]["train_s"] = round(train_s, 1)
    rec["extra"]["epochs_run"] = epochs_run
    rec["extra"]["best_val_rmse"] = round(best_rmse, 4)
    with torch.no_grad():
        pred_te_t, ms = timed_predict(
            lambda X: predict_rul(model, X), x_te, repeats=50)
        pred_te = pred_te_t.numpy()
        pred_va = predict_rul(model, x_va).numpy()
    rec["timing"]["infer_ms_per_row"] = round(ms, 5)
    rec["metrics"] = {
        "test_last_window_per_engine": evaluate_all(data.y_test_last, pred_te),
        "val_all_windows": evaluate_all(data.y_val, pred_va),
    }

    # Finish: BOTH comparison blocks (wave-2 global LSTM; wave-10
    # per-regime Ridge), in the wave-10 comparison shape.
    rec.pop("_wave2_model")
    base = _wave2_test_metrics(subset, "lstm")
    new = rec["metrics"]["test_last_window_per_engine"]
    rec["extra"]["comparison_wave2_global_normalisation"] = {
        "wave2_global": base,
        "wave12_perregime": {"rmse": new["rmse"], "mae": new["mae"],
                             "nasa_score": new["nasa_score"]},
        "delta_rmse": new["rmse"] - base["rmse"],
        "delta_mae": new["mae"] - base["mae"],
        "delta_nasa": new["nasa_score"] - base["nasa_score"],
        "delta_rmse_pct": 100.0 * (new["rmse"] / base["rmse"] - 1.0),
    }
    ridge = _read_record("cmapss_fd004__taskB__ridge_perregime__s42")
    rm = ridge["metrics"]["test_last_window_per_engine"]
    rec["extra"]["comparison_wave10_perregime_ridge"] = {
        "wave10_ridge_perregime": {
            "record": "cmapss_fd004__taskB__ridge_perregime__s42",
            "rmse": rm["rmse"], "mae": rm["mae"],
            "nasa_score": rm["nasa_score"]},
        "wave12_lstm_perregime": {"rmse": new["rmse"], "mae": new["mae"],
                                  "nasa_score": new["nasa_score"]},
        "delta_rmse": new["rmse"] - rm["rmse"],
        "delta_mae": new["mae"] - rm["mae"],
        "delta_nasa": new["nasa_score"] - rm["nasa_score"],
        "delta_rmse_pct": 100.0 * (new["rmse"] / rm["rmse"] - 1.0),
    }
    save_record(rec, {"test_rmse": round(new["rmse"], 3),
                      "test_mae": round(new["mae"], 3),
                      "test_nasa": round(new["nasa_score"], 1)})


# ------------------------------------------------------- order frames
# CWRU catalogue nominal speeds by motor load (CWRU Bearing Data
# Center, normal-baseline catalogue: 0 hp 1797 / 1 hp 1772 /
# 2 hp 1750 / 3 hp 1730 rpm). Used ONLY for the two recordings
# whose .mat files ship no RPM scalar; every other recording's own
# X<id>RPM variable is read from its file (all read 1796-1797 rpm,
# consistent with the catalogue's 0 hp nominal).
CWRU_CATALOGUE_RPM_BY_LOAD = {0: 1797.0, 1: 1772.0, 2: 1750.0, 3: 1730.0}


def _cwru_order_frame() -> tuple[pd.DataFrame, list[dict]]:
    """The wave-4 CWRU recording catalogue, featurised with the
    wave-12 combined (time + order) set. Window extraction mirrors
    run_wave9._cwru_spectral_frame exactly (same files, same X<id>
    variable selection, same non-overlapping 2048-sample windows);
    only the per-window feature function and the per-recording
    shaft speed differ."""
    from scipy.io import loadmat

    from platform_lib.adapters.cwru import SAMPLE_RATE_HZ, WINDOW_LEN

    rows, catalogue = [], []
    for fname, (condition, label) in CWRU_W4_RECORDINGS.items():
        path = RAW / "cwru" / fname
        if not path.exists():
            raise FileNotFoundError(f"CWRU recording missing: {path}")
        mat = loadmat(path)
        fid = int(fname.split(".")[0])
        key = f"X{fid:03d}_DE_time"
        if key not in mat:
            raise ValueError(f"{fname}: expected variable {key}, found "
                             f"{[k for k in mat if k.endswith('_DE_time')]}")
        series = np.asarray(mat[key], dtype=np.float64).ravel()
        rpm_key = f"X{fid:03d}RPM"
        if rpm_key in mat:
            rpm = float(np.asarray(mat[rpm_key]).ravel()[0])
            rpm_source = f"in-file variable {rpm_key}"
        else:
            load_hp = int(condition.rsplit("_", 1)[1])
            rpm = CWRU_CATALOGUE_RPM_BY_LOAD[load_hp]
            rpm_source = (f"CWRU catalogue nominal for {load_hp} hp "
                          f"load (this file ships no RPM scalar)")
        n_windows = len(series) // WINDOW_LEN
        for i in range(n_windows):
            feats = order_window_features(
                series[i * WINDOW_LEN:(i + 1) * WINDOW_LEN],
                SAMPLE_RATE_HZ, rpm, CWRU_6205)
            feats["label"] = label
            feats["recording"] = condition
            feats["source_file"] = fname
            feats["shaft_rpm"] = rpm
            rows.append(feats)
        catalogue.append({"file": fname, "condition": condition,
                          "label": label,
                          "n_samples": int(len(series)),
                          "n_windows": int(n_windows),
                          "shaft_rpm": rpm, "rpm_source": rpm_source,
                          "official_url": "https://engineering.case.edu/"
                                          f"sites/default/files/{fname}"})
    return pd.DataFrame(rows), catalogue


def _paderborn_order_frame() -> tuple[pd.DataFrame, dict]:
    """All wave-3/4-scope Paderborn windows with the wave-12
    combined set. Mirrors run_wave9._paderborn_spectral_frame
    (same bearings — the DEFAULT scope K001/KA01/KI01 — same
    vibration_1 channel, same 2048-sample windows); each window is
    read at its measurement's nominal shaft speed from the
    condition code (N15 = 1500 / N09 = 900 rpm, cross-checked
    against the recorded speed channel: 1499.6 / 899.7 rpm means)."""
    from platform_lib.adapters.cwru import WINDOW_LEN
    from platform_lib.adapters.paderborn import (DEFAULT_BEARINGS,
                                                 load_vibration,
                                                 parse_condition)

    rows, per_bearing = [], []
    raw_dir = RAW / "paderborn"
    for code in DEFAULT_BEARINGS:
        from platform_lib.adapters.paderborn import BEARINGS
        label, origin = BEARINGS[code]
        files = sorted((raw_dir / code).glob("*.mat"))
        if not files:
            raise FileNotFoundError(f"no .mat files under {raw_dir / code}")
        n_windows = 0
        for path in files:
            series = load_vibration(path)
            cond = parse_condition(path.name.split(f"_{code}_")[0])
            rpm = float(cond["speed_rpm_nominal"])
            for i in range(len(series) // WINDOW_LEN):
                feats = order_window_features(
                    series[i * WINDOW_LEN:(i + 1) * WINDOW_LEN],
                    64_000, rpm, PADERBORN_6203)
                feats.update(cond)
                feats["label"] = label
                feats["bearing"] = code
                feats["damage_origin"] = origin
                feats["source_file"] = path.name
                feats["shaft_rpm"] = rpm
                rows.append(feats)
                n_windows += 1
        per_bearing.append({"bearing": code, "label": label,
                            "damage_origin": origin,
                            "n_measurements": int(len(files)),
                            "n_windows": int(n_windows)})
    meta = {"bearings": per_bearing, "n_windows": int(len(rows))}
    return pd.DataFrame(rows), meta


def _order_physics_block() -> dict:
    return {
        "cwru": {**CWRU_6205.documentation(),
                 "sample_rate_hz": 12000,
                 "speeds": "per-recording: in-file X<id>RPM scalar "
                           "where the file ships one (10 of 12 "
                           "recordings, all 1796-1797 rpm); CWRU "
                           "catalogue nominal by load otherwise "
                           "(98.mat 1 hp -> 1772, 99.mat 2 hp -> 1750)",
                 "revolutions_per_window": 5.11,
                 "true_order_resolution": 0.196},
        "paderborn": {**PADERBORN_6203.documentation(),
                      "sample_rate_hz": 64000,
                      "speeds": "per-measurement nominal from the "
                                "condition code (N15 -> 1500 rpm, "
                                "N09 -> 900 rpm), cross-checked "
                                "against the recorded speed channel "
                                "(means 1499.6 / 899.7 rpm, std "
                                "0.18 rpm)",
                      "revolutions_per_window": {"N15": 0.80,
                                                 "N09": 0.48},
                      "true_order_resolution": {"N15": 1.25,
                                                "N09": 2.08}},
        "band_definition": "envelope-spectrum energy share within "
                           "+-2% of the target order (8x zero-padded "
                           "envelope spectrum; nearest single bin "
                           "if the band holds no bin centre); "
                           "normalised by total non-DC envelope-"
                           "spectrum energy",
    }


# ------------------------------------- 2i) CWRU recording-wise, order
def run_cwru_recordingwise_order() -> None:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (accuracy_score, classification_report,
                                 confusion_matrix, f1_score)

    from platform_lib.adapters.cwru import LABELS

    print("=== CWRU — Task A — RF, RECORDING-WISE (LOGO), combined "
          "order features ===", flush=True)
    df, catalogue = _cwru_order_frame()
    assert len(df) == 1122, f"CWRU frame drifted: {len(df)} != 1122"
    print(f"  windows: {len(df)} over {df['recording'].nunique()} "
          f"recordings; per class: {df['label'].value_counts().to_dict()}",
          flush=True)
    X = df[ORDER_COMBINED_FEATURE_COLUMNS]
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
    ir_idx = labels.index("InnerRace")

    def _cmp_block(rec_id: str) -> dict:
        r = _read_record(rec_id)
        m = r["metrics"]
        return {"accuracy": m["accuracy"], "f1_macro": m["f1_macro"],
                "innerrace_correct":
                    m["confusion_matrix"][ir_idx][ir_idx],
                "innerrace_support":
                    m["per_class"]["InnerRace"]["support"],
                "innerrace_f1": m["per_class"]["InnerRace"]["f1"],
                "record": rec_id}

    wave4 = _cmp_block("cwru__taskA__random_forest_recordingwise__s42")
    wave9 = _cmp_block(
        "cwru__taskA__random_forest_spectral_recordingwise__s42")
    this = {"accuracy": metrics["accuracy"],
            "f1_macro": metrics["f1_macro"],
            "innerrace_correct": cm[ir_idx][ir_idx],
            "innerrace_support":
                metrics["per_class"]["InnerRace"]["support"],
            "innerrace_f1": metrics["per_class"]["InnerRace"]["f1"]}
    comparison = {
        "wave4_time_domain_only": wave4,
        "wave9_combined_spectral": wave9,
        "wave12_combined_order": this,
        "delta_accuracy_vs_wave4": metrics["accuracy"] - wave4["accuracy"],
        "delta_accuracy_vs_wave9": metrics["accuracy"] - wave9["accuracy"],
        "delta_f1_macro_vs_wave4":
            metrics["f1_macro"] - wave4["f1_macro"],
        "delta_f1_macro_vs_wave9":
            metrics["f1_macro"] - wave9["f1_macro"],
    }
    rec = base_record(
        "cwru", "A", "random_forest_order_recordingwise",
        {"n_estimators": 300, "class_weight": "balanced", "n_jobs": -1,
         "features": FEATURE_PROTOCOL_NOTE})
    rec["split"] = {
        "protocol": "RECORDING-WISE leave-one-recording-out "
                    "(LeaveOneGroupOut over the SAME 12 recordings "
                    "as waves 4/9; groups = recording id): in every "
                    "fold the held-out recording's windows are the "
                    "test set and NO window from that recording "
                    "appears in training. Metrics are computed on "
                    "the pooled out-of-fold predictions. Protocol, "
                    "recordings, model config and seed are identical "
                    "to waves 4/9 — the ONLY change is the feature "
                    "set.",
        "n_recordings": int(df["recording"].nunique()),
        "n_windows": int(len(df)),
        "recordings_per_class": {k: int(v) for k, v in
                                 y.groupby(y).count().items()},
        "class_window_counts": {k: int(v) for k, v in
                                y.value_counts().items()},
    }
    rec["preprocessing"] = {
        "features": ORDER_COMBINED_FEATURE_COLUMNS,
        "feature_note": FEATURE_PROTOCOL_NOTE,
        "windowing": "non-overlapping windows of 2048 samples over "
                     "the DE channel of each recording (identical "
                     "to waves 2/4/9)",
        "recordings": catalogue,
    }
    rec["timing"] = {"train_s": round(total_train_s, 4),
                     "infer_ms_per_row": round(
                         total_pred_s / len(df) * 1000.0, 5)}
    rec["metrics"] = metrics
    rec["notes"] = (
        "Wave-12 counterpart of the wave-4/9 recording-wise "
        "records: identical protocol, recordings, model and seed; "
        "the only change is the feature set (10 time-domain -> 19 "
        "combined time + order-domain). The three-way comparison "
        "lives in extra.comparison_previous_feature_sets. "
        "Recording-wise evaluation still does NOT cover unseen "
        "fault diameters combined with unseen loads jointly, nor "
        "different machines. Data note (as wave 4): 99.mat as "
        "served bundles X098 alongside X099; the self-identified "
        "X099_DE_time series (2 hp normal) is the one used, read "
        "at the catalogue's 1750 rpm for its load (the file ships "
        "no RPM scalar).")
    rec["extra"] = {
        "per_recording_accuracy": per_recording,
        "comparison_previous_feature_sets": comparison,
        "order_physics_cwru": _order_physics_block()["cwru"],
    }
    save_record(rec, {"accuracy": round(metrics["accuracy"], 4),
                      "f1_macro": round(metrics["f1_macro"], 4),
                      "innerrace_correct": cm[ir_idx][ir_idx]})


# --------------------------------- 2ii/iii) transfers, order features
def _bearing_transfer_order(src: str, dst: str, frames) -> None:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (accuracy_score, classification_report,
                                 confusion_matrix, f1_score)

    print(f"=== Transfer — {src} RF -> {dst} (bearings, zero-shot, "
          f"combined order features) ===", flush=True)
    df_s, _ = frames[src]
    df_t, _ = frames[dst]
    X_s, y_s = df_s[ORDER_COMBINED_FEATURE_COLUMNS], df_s["label"]
    X_t, y_t = df_t[ORDER_COMBINED_FEATURE_COLUMNS], df_t["label"]
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

    def _baseline(rec_id: str) -> dict:
        r = _read_record(rec_id)
        m = r["metrics"]
        return {"accuracy": m["accuracy"], "f1_macro": m["f1_macro"],
                "normal_precision": m["per_class"]["Normal"]["precision"],
                "normal_recall": m["per_class"]["Normal"]["recall"],
                "normal_f1": m["per_class"]["Normal"]["f1"],
                "record": rec_id}

    def _deltas(base: dict) -> dict:
        return {"accuracy_delta":
                    metrics["accuracy"] - base["accuracy"],
                "f1_macro_delta":
                    metrics["f1_macro"] - base["f1_macro"],
                "normal_f1_delta":
                    metrics["per_class"]["Normal"]["f1"]
                    - base["normal_f1"]}

    wave5 = _baseline(f"{src}_to_{dst}__taskA__transfer_rf__s42")
    wave9 = _baseline(f"{src}_to_{dst}__taskA__transfer_rf_spectral__s42")
    in_domain = {
        "cwru": _read_record("cwru__taskA__random_forest_"
                             "recordingwise__s42"),
        "paderborn": _read_record("paderborn__taskA__random_forest__s42"),
    }
    rec = base_record(
        f"{src}_to_{dst}", "A", "transfer_rf_order",
        {"n_estimators": 300, "class_weight": "balanced", "n_jobs": -1,
         "features": FEATURE_PROTOCOL_NOTE,
         "training": f"fit on ALL {src} windows of the shared label "
                     f"set; no target data in training"})
    rec["experiment_id"] = f"{src}_to_{dst}__taskA__transfer_rf_order__s42"
    rec["split"] = {
        "protocol": f"zero-shot cross-dataset transfer: Random Forest "
                    f"trained on all {src} windows, tested on all "
                    f"{dst} windows; label set restricted to the "
                    f"shared {BEARING_LABELS}. Identical to the "
                    f"wave-5/9 transfer protocols — the ONLY change "
                    f"is the feature set (19 combined time + order).",
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
        "features": ORDER_COMBINED_FEATURE_COLUMNS,
        "feature_note": FEATURE_PROTOCOL_NOTE,
        "windowing": "non-overlapping 2048-sample windows in both "
                     "datasets — the window TIME SPAN still differs "
                     "by sampling rate: 2048 samples = 170.7 ms at "
                     "CWRU's 12 kHz vs 32.0 ms at Paderborn's "
                     "64 kHz, i.e. 5.11 shaft revolutions vs 0.80 "
                     "(N15) / 0.48 (N09). True order resolution is "
                     "therefore 0.196 orders on CWRU but 1.25/2.08 "
                     "orders on Paderborn — coarser than the "
                     "spacing between the characteristic orders "
                     "themselves; the order features on Paderborn "
                     "windows are blunt by construction of the "
                     "shared 2048-sample protocol.",
        "excluded": "CWRU 'Ball' class excluded (Paderborn scope "
                    "has no ball-fault bearing); CWRU windows of "
                    "that class are dropped from BOTH training "
                    "(when CWRU is the source) and scoring",
        "scaling": "none (tree model, as in the wave-5/9 records)",
    }
    rec["metrics"] = metrics
    rec["timing"] = {"train_s": round(train_s, 4),
                     "infer_ms_per_row": round(ms, 5)}
    rec["notes"] = (
        f"Zero-shot {src} -> {dst} bearing-fault transfer with the "
        f"wave-12 order-domain feature set — the wave-5/9 transfer "
        f"re-run with the feature set as the only changed variable. "
        f"The order features align the characteristic defect "
        f"frequencies of the two DIFFERENT bearings (6205 vs 6203) "
        f"in shaft-speed-relative units, which neither the "
        f"time-domain nor the rate-free spectral sets attempted. "
        f"What remains unaligned, documented rather than "
        f"corrected: rig transfer functions, sensor placement and "
        f"mounting; load regimes; fault sizes/geometries; and — "
        f"binding on the Paderborn side — a 2048-sample window "
        f"holds less than one shaft revolution, so the true order "
        f"resolution there (1.25-2.08 orders) is coarser than the "
        f"order spacing. Both earlier baselines are carried in "
        f"extra; the result is reported as measured, whether or "
        f"not it improves.")
    rec["extra"] = {
        "wave5_time_domain_baseline": wave5,
        "wave9_spectral_baseline": wave9,
        "delta_vs_wave5_time_domain": _deltas(wave5),
        "delta_vs_wave9_spectral": _deltas(wave9),
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
        "order_physics": _order_physics_block(),
    }
    save_record(rec, {"accuracy": round(metrics["accuracy"], 4),
                      "f1_macro": round(metrics["f1_macro"], 4),
                      "normal_f1": round(
                          metrics["per_class"]["Normal"]["f1"], 4)})


def run_bearing_transfers_order() -> None:
    cwru_df, catalogue = _cwru_order_frame()
    pad_df, pad_meta = _paderborn_order_frame()
    assert len(cwru_df) == 1122 and len(pad_df) == 30022, (
        f"frame drift: cwru {len(cwru_df)} (want 1122), "
        f"paderborn {len(pad_df)} (want 30022)")
    frames = {
        "cwru": (cwru_df[cwru_df["label"].isin(BEARING_LABELS)],
                 catalogue),
        "paderborn": (pad_df[pad_df["label"].isin(BEARING_LABELS)],
                      pad_meta),
    }
    assert len(frames["cwru"][0]) == 945
    _bearing_transfer_order("cwru", "paderborn", frames)
    _bearing_transfer_order("paderborn", "cwru", frames)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", default="all",
                        choices=["fd004-lstm", "cwru", "transfers", "all"])
    args = parser.parse_args()
    if args.set in ("fd004-lstm", "all"):
        run_fd004_lstm_perregime()
    if args.set in ("cwru", "all"):
        run_cwru_recordingwise_order()
    if args.set in ("transfers", "all"):
        run_bearing_transfers_order()
