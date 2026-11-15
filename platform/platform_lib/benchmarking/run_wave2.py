"""PREDICAI platform — benchmarking wave 2 (Phases 4/6/7 continuation).

Wave 1 (run_benchmark.py) covered AI4I, SECOM and FD003. This module adds:

- FD002 / FD004 Task B (Ridge baseline + shipped-recipe LSTM), using the
  same per-subset feature-derivation rule as FD003 (rul_pipeline.py).
- Phase 6 cross-dataset transfer: the SHIPPED FD001 LSTM
  (ml/artifacts/best_model.pt) scored zero-shot on FD003's official test
  set, using FD001's exact 14-sensor feature set and FD001's scaler
  statistics from ml/artifacts/preprocessing.json. Nothing is fitted on
  FD003 — that is what makes it a transfer measurement rather than a
  second in-domain model.
- Phase 7 SHAP: TreeExplainer attributions for a retrained AI4I LightGBM
  (identical config/split to the wave-1 record), stored as an
  explainability record and compared against the wave-1 permutation
  importance ranking (which was computed on the Random Forest — a
  different model, so the comparison is method-vs-method, stated as such).
- CWRU Task A: Random Forest on window statistics from official CWRU
  12 kHz drive-end .mat recordings (see adapters/cwru.py).

Usage (from the platform/ directory):
    ../.venv/bin/python -m platform_lib.benchmarking.run_wave2 --set fd002
    ../.venv/bin/python -m platform_lib.benchmarking.run_wave2 --set fd004
    ../.venv/bin/python -m platform_lib.benchmarking.run_wave2 --set transfer
    ../.venv/bin/python -m platform_lib.benchmarking.run_wave2 --set shap
    ../.venv/bin/python -m platform_lib.benchmarking.run_wave2 --set cwru

Seed is 42 everywhere, matching wave 1.
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
    classification_metrics,
    save_record,
    timed_predict,
)

LSTM_CFG = dict(hidden_1=128, hidden_2=64, head_dim=32, dropout=0.2,
                epochs=60, batch_size=64, lr=1e-3, weight_decay=1e-4,
                grad_clip=1.0, patience=10, seed=SEED)


# ------------------------------------------------------- C-MAPSS subsets
def _subset_data(subset: str):
    from platform_lib.adapters.cmapss import CMapssAdapter
    from platform_lib.rul_pipeline import build_rul_data

    adapter = CMapssAdapter(RAW / "cmapss" / "extracted", subset=subset)
    train, meta = adapter.load()
    test, rul = adapter.load_test()
    data = build_rul_data(train, test, rul, cap=125, seq_len=30, seed=SEED)
    return meta, data


def _subset_record(subset: str, model: str, params: dict, meta: dict,
                   data) -> dict:
    key = f"cmapss_{subset.lower()}"
    rec = base_record(key, "B", model, params)
    rec["split"] = {
        "protocol": "engine-wise 80/20 train/val (seed 42) + official "
                    "provided test set, last window per engine",
        "n_train_engines": len(data.train_engines),
        "n_val_engines": len(data.val_engines),
        "n_train_windows": int(len(data.X_train)),
        "n_val_windows": int(len(data.X_val)),
        "n_test_engines": int(len(data.X_test_last)),
    }
    rec["preprocessing"] = {
        "features": data.features,
        "n_features": len(data.features),
        "feature_derivation": "candidates = 3 settings + 21 sensors; dropped "
                              "train-split std < 0.005 (see rul_pipeline.py)",
        "dropped_candidates": data.dropped,
        "rul_cap": 125, "seq_len": 30,
        "scaling": "z-score (ddof=0) fit on train engines only",
    }
    kept_settings = [f for f in data.features if f.startswith("setting_")]
    dropped_cols = [d["column"] for d in data.dropped]
    rec["notes"] = (
        f"{subset} = {meta['operating_conditions']} operating conditions, "
        f"{meta['fault_modes']} fault mode(s). Feature set derived from "
        f"{subset}'s own train split by the std<0.005 rule: settings kept = "
        f"{kept_settings if kept_settings else 'none'}; dropped = "
        f"{dropped_cols}. " + meta.get("data_nature", ""))
    rec["extra"] = {"candidate_train_stds": data.candidate_stds}
    return rec


def run_subset_ridge(subset: str) -> None:
    from sklearn.linear_model import Ridge

    from predicai_ml.evaluate import evaluate_all

    print(f"=== C-MAPSS {subset} — Task B — Ridge (flattened windows) ===",
          flush=True)
    meta, data = _subset_data(subset)
    print(f"  features ({len(data.features)}): {data.features}", flush=True)
    print(f"  dropped: {[d['column'] for d in data.dropped]}", flush=True)
    Xtr = data.X_train.reshape(len(data.X_train), -1)
    Xva = data.X_val.reshape(len(data.X_val), -1)
    Xte = data.X_test_last.reshape(len(data.X_test_last), -1)

    rec = _subset_record(subset, "ridge_flat",
                         {"alpha": 1.0,
                          "input": "flattened (30 x F) windows, "
                                   "predictions clipped at >= 0"},
                         meta, data)
    model = Ridge(alpha=1.0)
    t0 = time.perf_counter()
    model.fit(Xtr, data.y_train)
    rec["timing"]["train_s"] = round(time.perf_counter() - t0, 4)
    pred_te, ms = timed_predict(
        lambda X: np.clip(model.predict(X), 0, None), Xte, repeats=50)
    rec["timing"]["infer_ms_per_row"] = round(ms, 5)
    pred_va = np.clip(model.predict(Xva), 0, None)
    rec["metrics"] = {
        "test_last_window_per_engine": evaluate_all(data.y_test_last, pred_te),
        "val_all_windows": evaluate_all(data.y_val, pred_va),
    }
    save_record(rec, {"test_rmse": round(rec["metrics"]["test_last_window_per_engine"]["rmse"], 3),
                      "test_mae": round(rec["metrics"]["test_last_window_per_engine"]["mae"], 3),
                      "test_nasa": round(rec["metrics"]["test_last_window_per_engine"]["nasa_score"], 1)})


def run_subset_lstm(subset: str) -> None:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    from predicai_ml.evaluate import evaluate_all
    from predicai_ml.model import LSTMRULRegressor, predict_rul

    print(f"=== C-MAPSS {subset} — Task B — LSTM (local CPU) ===", flush=True)
    meta, data = _subset_data(subset)
    print(f"  features ({len(data.features)}): {data.features}", flush=True)
    print(f"  train windows {len(data.X_train)} / val {len(data.X_val)}",
          flush=True)
    cfg = dict(LSTM_CFG)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    rec = _subset_record(subset, "lstm",
                         {**cfg, "device": "cpu (local run)",
                          "architecture": "LSTM(128)->LSTM(64)->head 32->1 "
                                          "(same as shipped FD001 model)"},
                         meta, data)

    ds = TensorDataset(torch.from_numpy(data.X_train),
                       torch.from_numpy(data.y_train))
    dl = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True)
    x_va = torch.from_numpy(data.X_val)
    x_te = torch.from_numpy(data.X_test_last)
    y_va = torch.from_numpy(data.y_val)

    model = LSTMRULRegressor(len(data.features), cfg["hidden_1"],
                             cfg["hidden_2"], cfg["head_dim"], cfg["dropout"])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                            weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
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
        print(f"  {subset.lower()}-lstm epoch {epoch:02d}  val RMSE "
              f"{val_rmse:.3f}  ({(time.time() - t0) / 60:.1f} min)",
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
    save_record(rec, {"test_rmse": round(rec["metrics"]["test_last_window_per_engine"]["rmse"], 3),
                      "test_mae": round(rec["metrics"]["test_last_window_per_engine"]["mae"], 3),
                      "test_nasa": round(rec["metrics"]["test_last_window_per_engine"]["nasa_score"], 1)})


# ------------------------------------------------- Phase 6: transfer run
def run_transfer_fd001_to_fd003() -> None:
    import torch

    from predicai_ml.evaluate import evaluate_all
    from predicai_ml.model import LSTMRULRegressor, predict_rul
    from predicai_ml.preprocessing import (Scaler, add_test_rul, clip_rul,
                                            last_window_per_engine)
    from platform_lib.adapters.cmapss import CMapssAdapter

    print("=== Phase 6 transfer — shipped FD001 LSTM -> FD003 (zero-shot) ===",
          flush=True)
    prep = json.loads((REPO_ROOT / "ml" / "artifacts" /
                       "preprocessing.json").read_text())
    features = prep["features"]
    scaler = Scaler(features=list(features),
                    mean=np.array(prep["scaler"]["mean"], dtype=np.float64),
                    scale=np.array(prep["scaler"]["scale"], dtype=np.float64))

    adapter = CMapssAdapter(RAW / "cmapss" / "extracted", subset="FD003")
    test, rul = adapter.load_test()
    test = clip_rul(add_test_rul(test, rul), cap=prep["rul_cap"])
    test_scaled = scaler.transform(test)
    X_te, y_te, engines = last_window_per_engine(
        test_scaled, features, prep["seq_len"])
    print(f"  FD003 test last windows: {X_te.shape}; features: {features}",
          flush=True)

    model = LSTMRULRegressor(len(features), 128, 64, 32, 0.2)
    state = torch.load(REPO_ROOT / "ml" / "artifacts" / "best_model.pt",
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
    fd003_indomain = json.loads(
        (PLATFORM_DIR / "experiments" /
         "cmapss_fd003__taskB__lstm__s42.json").read_text())[
        "metrics"]["test_last_window_per_engine"]

    rec = base_record(
        "cmapss_fd001_to_fd003", "B", "transfer_lstm",
        {"architecture": "LSTM(128)->LSTM(64)->head 32->1 (shipped FD001 "
                         "model, weights unchanged)",
         "weights": "ml/artifacts/best_model.pt (owner Kaggle GPU run)",
         "training": "none — zero-shot transfer, no FD003 fitting of any "
                     "kind (no fine-tuning, no scaler fitting)"})
    rec["experiment_id"] = "cmapss_fd001_to_fd003__taskB__transfer_lstm__s42"
    rec["split"] = {
        "protocol": "zero-shot cross-dataset transfer: shipped FD001 model "
                    "applied to the official FD003 test set, last window "
                    "per engine; FD003 data used for scoring only",
        "n_test_engines": int(len(X_te)),
        "source_dataset": "cmapss_fd001",
        "target_dataset": "cmapss_fd003",
    }
    rec["preprocessing"] = {
        "features": features,
        "n_features": len(features),
        "feature_set": "FD001's exact 14 sensors from "
                       "ml/artifacts/preprocessing.json (sensor_6 excluded, "
                       "as the shipped model expects)",
        "scaling": "z-score with FD001 train-split statistics from "
                   "ml/artifacts/preprocessing.json — source-fitted; no "
                   "FD003 statistics used anywhere",
        "rul_cap": prep["rul_cap"], "seq_len": prep["seq_len"],
    }
    rec["metrics"] = {"test_last_window_per_engine": metrics}
    # No training happens in a transfer run; train_s is recorded as 0.0 so
    # the schema stays uniform, and this note is the authoritative reading.
    rec["timing"] = {"train_s": 0.0, "infer_ms_per_row": round(ms, 5)}
    rec["notes"] = (
        "Zero-shot FD001 -> FD003 transfer. FD003 adds a second fault mode "
        "(fan) that FD001 training never contains, and FD003 sensor "
        "distributions differ; degradation vs both in-domain references is "
        "the expected and reportable result. train_s = 0.0 because no "
        "training was performed in this run.")
    rec["extra"] = {
        "reference_fd001_in_domain": {
            "rmse": fd001_ref["rmse"], "mae": fd001_ref["mae"],
            "nasa_score": fd001_ref["nasa_score"],
            "source": "ml/artifacts/metrics.json (shipped model on FD001 "
                      "test, same last-window protocol)"},
        "reference_fd003_in_domain_lstm": {
            "rmse": fd003_indomain["rmse"], "mae": fd003_indomain["mae"],
            "nasa_score": fd003_indomain["nasa_score"],
            "source": "platform record cmapss_fd003__taskB__lstm__s42 "
                      "(local CPU run, FD003-native 15 features)"},
        "degradation_vs_fd003_in_domain": {
            "rmse_delta": metrics["rmse"] - fd003_indomain["rmse"],
            "rmse_pct_increase": 100.0 * (metrics["rmse"] /
                                          fd003_indomain["rmse"] - 1.0),
            "mae_delta": metrics["mae"] - fd003_indomain["mae"],
            "nasa_delta": metrics["nasa_score"] - fd003_indomain["nasa_score"],
        },
        "degradation_vs_fd001_in_domain": {
            "rmse_delta": metrics["rmse"] - fd001_ref["rmse"],
            "rmse_pct_increase": 100.0 * (metrics["rmse"] /
                                          fd001_ref["rmse"] - 1.0),
        },
        "per_engine": [
            {"engine": int(e), "y_true": float(t), "y_pred": float(p)}
            for e, t, p in zip(engines, y_te, pred)
        ],
    }
    save_record(rec, {"test_rmse": round(metrics["rmse"], 3),
                      "test_mae": round(metrics["mae"], 3),
                      "test_nasa": round(metrics["nasa_score"], 1)})


# ------------------------------------------------------- Phase 7: SHAP
def run_shap_ai4i() -> None:
    import shap
    from scipy.stats import spearmanr
    from sklearn.model_selection import train_test_split

    from platform_lib.adapters.ai4i import AI4IAdapter

    print("=== AI4I 2020 — Task A — SHAP (LightGBM, TreeExplainer) ===",
          flush=True)
    df, meta = AI4IAdapter(RAW / "ai4i").load()
    X = df[meta["numeric_features"]].copy()
    X = pd.concat([X, pd.get_dummies(df["Type"], prefix="Type", dtype=float)],
                  axis=1)
    y = df[meta["target"]].astype(int)
    feature_names = list(X.columns)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)
    neg, pos = int((y_train == 0).sum()), int((y_train == 1).sum())

    import lightgbm as lgb
    est = lgb.LGBMClassifier(n_estimators=300, class_weight="balanced",
                             random_state=SEED, n_jobs=-1, verbose=-1)
    X_tr, X_te = X_train.to_numpy(), X_test.to_numpy()
    t0 = time.perf_counter()
    est.fit(X_tr, y_train)
    train_s = time.perf_counter() - t0
    y_pred, ms = timed_predict(est.predict, X_te)
    y_score = est.predict_proba(X_te)[:, 1]
    metrics = classification_metrics(y_test, y_pred, y_score)

    t0 = time.perf_counter()
    explainer = shap.TreeExplainer(est)
    sv = explainer.shap_values(X_te)
    shap_s = time.perf_counter() - t0
    if isinstance(sv, list):          # older shap: [class0, class1]
        sv = sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3:                  # (n, features, classes)
        sv = sv[:, :, 1]
    mean_abs = np.abs(sv).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    ranking = [{"feature": feature_names[i],
                "mean_abs_shap": float(mean_abs[i])} for i in order]

    # Wave-1 permutation importance (Random Forest) for the comparison.
    rf_rec = json.loads((PLATFORM_DIR / "experiments" /
                         "ai4i2020__taskA__random_forest__s42.json").read_text())
    perm = rf_rec["extra"]["permutation_importance_test_f1"]
    perm_rank = {p["feature"]: r + 1 for r, p in enumerate(perm)}
    shap_rank = {r["feature"]: i + 1 for i, r in enumerate(ranking)}
    common = [f for f in feature_names if f in perm_rank and f in shap_rank]
    rho, pval = spearmanr([perm_rank[f] for f in common],
                          [shap_rank[f] for f in common])
    top5_shap = [r["feature"] for r in ranking[:5]]
    top5_perm = [p["feature"] for p in perm[:5]]

    rec = base_record("ai4i2020", "A", "shap_lightgbm",
                      {"model": "LGBMClassifier(n_estimators=300, "
                                "class_weight='balanced') — identical "
                                "config to the wave-1 lightgbm record",
                       "explainer": "shap.TreeExplainer",
                       "shap_version": shap.__version__,
                       "explained_class": "1 (Machine failure)",
                       "n_explained_rows": int(len(X_te))})
    rec["split"] = {
        "protocol": "stratified 80/20, seed 42 (identical to wave-1 AI4I "
                    "records)",
        "n_train": int(len(X_train)), "n_test": int(len(X_test)),
        "train_failures": pos, "test_failures": int(y_test.sum()),
    }
    rec["preprocessing"] = {
        "features": feature_names,
        "excluded_id_columns": meta["id_columns"],
        "excluded_leakage_columns": meta["leakage_columns"],
        "categorical_encoding": "one-hot of Type (fixed domain levels L/M/H)",
    }
    rec["metrics"] = metrics
    rec["timing"] = {"train_s": round(train_s, 4),
                     "infer_ms_per_row": round(ms, 5)}
    rec["notes"] = (
        "Explainability record (Phase 7): the LightGBM model was retrained "
        "with the exact wave-1 config and split so the attributions belong "
        "to a real, scored model; `metrics` are that retrained model's "
        "test metrics. SHAP values are TreeExplainer attributions for the "
        "failure class on the model's raw (log-odds) output scale.")
    rec["extra"] = {
        "shap_mean_abs_ranking": ranking,
        "shap_compute_s": round(shap_s, 3),
        "comparison_with_permutation_importance": {
            "permutation_source": "wave-1 record ai4i2020__taskA__"
                                  "random_forest__s42 (Random Forest, "
                                  "test-F1 permutation importance)",
            "caveat": "different model (RF vs LightGBM) and different "
                      "method (permutation vs SHAP) — this compares "
                      "rankings, not identical quantities",
            "top5_shap": top5_shap,
            "top5_permutation": top5_perm,
            "top5_overlap": sorted(set(top5_shap) & set(top5_perm)),
            "spearman_rho_all_features": float(rho),
            "spearman_p_value": float(pval),
            "permutation_ranking": [
                {"feature": p["feature"],
                 "importance_mean": p["importance_mean"]} for p in perm],
        },
    }
    save_record(rec, {"f1": round(metrics["f1"], 4),
                      "top_feature": ranking[0]["feature"],
                      "top5_overlap_with_permutation": len(
                          set(top5_shap) & set(top5_perm))})


# ------------------------------------------------------------------ CWRU
def run_cwru() -> None:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (accuracy_score, classification_report,
                                 confusion_matrix, f1_score, precision_score,
                                 recall_score)
    from sklearn.model_selection import train_test_split

    from platform_lib.adapters.cwru import LABELS, CWRUAdapter

    print("=== CWRU — Task A — Random Forest on window statistics ===",
          flush=True)
    df, meta = CWRUAdapter(RAW / "cwru").load()
    feature_cols = meta["feature_columns"]
    X, y = df[feature_cols], df["label"]
    print(f"  windows: {len(df)}; per class: "
          f"{y.value_counts().to_dict()}", flush=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)
    est = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                 random_state=SEED, n_jobs=-1)
    rec = base_record("cwru", "A", "random_forest",
                      {"n_estimators": 300, "class_weight": "balanced",
                       "n_jobs": -1,
                       "features": "per-window time-domain statistics of "
                                   "the drive-end vibration series"})
    rec["split"] = {
        "protocol": "stratified 80/20 over WINDOWS, seed 42 — windows from "
                    "the same recording appear in both splits (see notes; "
                    "this inflates the score vs a recording-wise split)",
        "n_train": int(len(X_train)), "n_test": int(len(X_test)),
        "class_counts_total": {k: int(v) for k, v in
                               y.value_counts().items()},
    }
    rec["preprocessing"] = {
        "features": feature_cols,
        "windowing": f"non-overlapping windows of {meta['window_len']} "
                     "samples over the DE channel of each recording",
        "recordings": meta["recordings"],
    }
    rec["notes"] = (
        "LIMITATION (binding): the split is over windows, not recordings — "
        "adjacent windows of one recording share its fault signature, so "
        "these scores are optimistic relative to a recording-wise split and "
        "must not be quoted as generalisation performance. Only 4 "
        "recordings (one per class, 0 hp load, 0.007-inch faults) are "
        "integrated in this wave. " + meta.get("data_nature", ""))
    t0 = time.perf_counter()
    est.fit(X_train, y_train)
    rec["timing"]["train_s"] = round(time.perf_counter() - t0, 4)
    y_pred, ms = timed_predict(est.predict, X_test)
    rec["timing"]["infer_ms_per_row"] = round(ms, 5)
    labels = [c for c in LABELS if c in set(y)]
    report = classification_report(y_test, y_pred, labels=labels,
                                   output_dict=True, zero_division=0)
    rec["metrics"] = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision_macro": float(precision_score(
            y_test, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(
            y_test, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro",
                                   zero_division=0)),
        "confusion_matrix": confusion_matrix(
            y_test, y_pred, labels=labels).tolist(),
        "confusion_labels": labels,
        "per_class": {c: {"precision": report[c]["precision"],
                          "recall": report[c]["recall"],
                          "f1": report[c]["f1-score"],
                          "support": int(report[c]["support"])}
                      for c in labels},
    }
    save_record(rec, {"accuracy": round(rec["metrics"]["accuracy"], 4),
                      "f1_macro": round(rec["metrics"]["f1_macro"], 4)})


# ------------------------------------------------------------------- CLI
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", required=True,
                        choices=["fd002", "fd004", "fd002-ridge",
                                 "fd004-ridge", "fd002-lstm", "fd004-lstm",
                                 "transfer", "shap", "cwru"])
    args = parser.parse_args()
    if args.set == "fd002":
        run_subset_ridge("FD002")
        run_subset_lstm("FD002")
    elif args.set == "fd004":
        run_subset_ridge("FD004")
        run_subset_lstm("FD004")
    elif args.set == "fd002-ridge":
        run_subset_ridge("FD002")
    elif args.set == "fd004-ridge":
        run_subset_ridge("FD004")
    elif args.set == "fd002-lstm":
        run_subset_lstm("FD002")
    elif args.set == "fd004-lstm":
        run_subset_lstm("FD004")
    elif args.set == "transfer":
        run_transfer_fd001_to_fd003()
    elif args.set == "shap":
        run_shap_ai4i()
    elif args.set == "cwru":
        run_cwru()
    print("done.", flush=True)


if __name__ == "__main__":
    main()
