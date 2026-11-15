"""PREDICAI platform — benchmarking engine (ARCHITECTURE.md §4).

Runs named experiment sets and appends one JSON record per experiment to
platform/experiments/, maintaining experiments/INDEX.json. No number is
reported anywhere in the platform except from these records.

Usage (from the platform/ directory):
    ../.venv/bin/python -m platform_lib.benchmarking.run_benchmark --set ai4i
    ../.venv/bin/python -m platform_lib.benchmarking.run_benchmark --set secom
    ../.venv/bin/python -m platform_lib.benchmarking.run_benchmark --set fd003
    ../.venv/bin/python -m platform_lib.benchmarking.run_benchmark --set all

Seed is 42 everywhere. Splits: stratified 80/20 (tabular); engine-wise
80/20 + official test set (C-MAPSS). All fitted preprocessing statistics
come from training data only.
"""
from __future__ import annotations

import argparse
import json
import platform as _platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
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

RAW = PLATFORM_DIR / "data" / "raw"
EXPERIMENTS_DIR = PLATFORM_DIR / "experiments"
SEED = 42


# --------------------------------------------------------------- record core
def _lib_versions() -> dict:
    versions = {}
    for pkg in ("numpy", "pandas", "scikit-learn", "scipy", "torch",
                "xgboost", "lightgbm"):
        try:
            versions[pkg] = importlib_metadata.version(pkg)
        except importlib_metadata.PackageNotFoundError:
            versions[pkg] = None
    return versions


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def base_record(dataset: str, task: str, model: str, model_params: dict) -> dict:
    return {
        "experiment_id": f"{dataset}__task{task}__{model}__s{SEED}",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": dataset,
        "task": task,
        "model": model,
        "model_params": model_params,
        "seed": SEED,
        "split": {},
        "preprocessing": {},
        "metrics": {},
        "timing": {},
        "environment": {
            "python": _platform.python_version(),
            "platform": _platform.platform(),
            "libraries": _lib_versions(),
        },
        "git_commit": _git_commit(),
        "notes": "",
        "extra": {},
    }


def save_record(record: dict, headline: dict) -> Path:
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    path = EXPERIMENTS_DIR / f"{record['experiment_id']}.json"
    path.write_text(json.dumps(record, indent=2, default=float))
    index_path = EXPERIMENTS_DIR / "INDEX.json"
    index = (json.loads(index_path.read_text()) if index_path.exists()
             else {"experiments": []})
    entry = {
        "experiment_id": record["experiment_id"],
        "dataset": record["dataset"],
        "task": record["task"],
        "model": record["model"],
        "file": path.name,
        "timestamp_utc": record["timestamp_utc"],
        "headline": headline,
    }
    index["experiments"] = [e for e in index["experiments"]
                            if e["experiment_id"] != record["experiment_id"]]
    index["experiments"].append(entry)
    index_path.write_text(json.dumps(index, indent=2, default=float))
    print(f"  saved {path.name}  headline={headline}", flush=True)
    return path


# ------------------------------------------------------------------ helpers
def classification_metrics(y_true, y_pred, y_score=None) -> dict:
    from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                                 precision_score, recall_score, roc_auc_score)
    m = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    if y_score is not None:
        m["roc_auc"] = float(roc_auc_score(y_true, y_score))
    return m


def timed_predict(predict_fn, X, repeats: int = 10):
    """Return (predictions, wall-clock ms per row, averaged over repeats)."""
    preds = predict_fn(X)
    t0 = time.perf_counter()
    for _ in range(repeats):
        preds = predict_fn(X)
    dt = time.perf_counter() - t0
    return preds, (dt / repeats) / len(X) * 1000.0


# --------------------------------------------------------------------- AI4I
def run_ai4i() -> None:
    from sklearn.ensemble import (HistGradientBoostingClassifier,
                                  RandomForestClassifier)
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    from platform_lib.adapters.ai4i import AI4IAdapter

    print("=== AI4I 2020 — Task A ===", flush=True)
    df, meta = AI4IAdapter(RAW / "ai4i").load()
    X = df[meta["numeric_features"]].copy()
    X = pd.concat([X, pd.get_dummies(df["Type"], prefix="Type", dtype=float)],
                  axis=1)
    y = df[meta["target"]].astype(int)
    feature_names = list(X.columns)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)
    baseline_acc = float((y_test == 0).mean())  # all-negative predictor
    neg, pos = int((y_train == 0).sum()), int((y_train == 1).sum())
    print(f"  train {len(X_train)} / test {len(X_test)}; "
          f"train failures {pos}; all-negative baseline acc {baseline_acc:.4f}",
          flush=True)

    balanced_w = np.where(y_train == 1, len(y_train) / (2 * pos),
                          len(y_train) / (2 * neg))
    # 4th tuple element: feed numpy arrays instead of a DataFrame. XGBoost
    # rejects AI4I's bracketed column names ("Torque [Nm]") as feature
    # names; feature identity is preserved via `feature_names` in the record.
    models = {
        "logistic_regression": (
            Pipeline([("scaler", StandardScaler()),
                      ("clf", LogisticRegression(
                          max_iter=1000, class_weight="balanced",
                          random_state=SEED))]),
            {"pipeline": "StandardScaler + LogisticRegression",
             "max_iter": 1000, "class_weight": "balanced"}, None, False),
        "random_forest": (
            RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                   random_state=SEED, n_jobs=-1),
            {"n_estimators": 300, "class_weight": "balanced", "n_jobs": -1},
            None, False),
        "hist_gradient_boosting": (
            HistGradientBoostingClassifier(random_state=SEED),
            {"random_state": SEED,
             "imbalance_handling": "balanced sample_weight at fit time"},
            balanced_w, False),
        "xgboost": (
            __import__("xgboost").XGBClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.1,
                scale_pos_weight=neg / pos, eval_metric="logloss",
                random_state=SEED, n_jobs=-1),
            {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.1,
             "scale_pos_weight": round(neg / pos, 3)}, None, True),
        "lightgbm": (
            __import__("lightgbm").LGBMClassifier(
                n_estimators=300, class_weight="balanced",
                random_state=SEED, n_jobs=-1, verbose=-1),
            {"n_estimators": 300, "class_weight": "balanced"}, None, True),
    }

    for name, (est, params, sample_weight, numpy_input) in models.items():
        X_tr = X_train.to_numpy() if numpy_input else X_train
        X_te = X_test.to_numpy() if numpy_input else X_test
        rec = base_record("ai4i2020", "A", name, params)
        rec["split"] = {
            "protocol": "stratified 80/20, seed 42",
            "n_train": int(len(X_train)), "n_test": int(len(X_test)),
            "train_failures": pos, "test_failures": int(y_test.sum()),
            "all_negative_baseline_accuracy": baseline_acc,
        }
        rec["preprocessing"] = {
            "features": feature_names,
            "excluded_id_columns": meta["id_columns"],
            "excluded_leakage_columns": meta["leakage_columns"],
            "categorical_encoding": "one-hot of Type (fixed domain levels L/M/H)",
            "scaling": "StandardScaler inside pipeline (logistic regression only)",
        }
        rec["notes"] = ("Synthetic dataset; failure modes are deterministic "
                        "rules of the features, so high scores reflect the "
                        "rule structure, not real-world machine performance. "
                        "Failure-mode flag columns excluded (target leakage).")
        t0 = time.perf_counter()
        if sample_weight is not None:
            est.fit(X_tr, y_train, sample_weight=sample_weight)
        else:
            est.fit(X_tr, y_train)
        rec["timing"]["train_s"] = round(time.perf_counter() - t0, 4)
        y_pred, ms = timed_predict(est.predict, X_te)
        rec["timing"]["infer_ms_per_row"] = round(ms, 5)
        y_score = est.predict_proba(X_te)[:, 1]
        rec["metrics"] = classification_metrics(y_test, y_pred, y_score)
        if name == "random_forest":
            from sklearn.inspection import permutation_importance
            pi = permutation_importance(est, X_test, y_test, n_repeats=10,
                                        random_state=SEED, scoring="f1")
            order = np.argsort(pi.importances_mean)[::-1]
            rf_record_extra = {
                "permutation_importance_test_f1": [
                    {"feature": feature_names[i],
                     "importance_mean": float(pi.importances_mean[i]),
                     "importance_std": float(pi.importances_std[i])}
                    for i in order
                ]
            }
            rec["extra"] = rf_record_extra
        save_record(rec, {"f1": round(rec["metrics"]["f1"], 4),
                          "roc_auc": round(rec["metrics"]["roc_auc"], 4),
                          "recall": round(rec["metrics"]["recall"], 4)})


# -------------------------------------------------------------------- SECOM
def run_secom() -> None:
    from sklearn.ensemble import IsolationForest, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    from platform_lib.adapters.secom import SECOMAdapter

    print("=== SECOM — Tasks A + C ===", flush=True)
    df, meta = SECOMAdapter(RAW / "secom").load()
    feature_cols = meta["feature_columns"]
    y = df["target"]
    idx_train, idx_test = train_test_split(
        df.index, test_size=0.2, stratify=y, random_state=SEED)
    train_df, test_df = df.loc[idx_train], df.loc[idx_test]
    y_train, y_test = y.loc[idx_train], y.loc[idx_test]

    # Fitted preprocessing — TRAIN split only (missingness filter, constant
    # filter, imputation medians, scaler).
    miss_frac = train_df[feature_cols].isna().mean()
    keep = [c for c in feature_cols if miss_frac[c] <= 0.5]
    n_after_missing = len(keep)
    keep = [c for c in keep if train_df[c].nunique(dropna=True) > 1]
    medians = train_df[keep].median()
    X_train = train_df[keep].fillna(medians)
    X_test = test_df[keep].fillna(medians)
    scaler = StandardScaler().fit(X_train)
    X_train_s = pd.DataFrame(scaler.transform(X_train), columns=keep,
                             index=X_train.index)
    X_test_s = pd.DataFrame(scaler.transform(X_test), columns=keep,
                            index=X_test.index)
    print(f"  features: {len(feature_cols)} raw -> {n_after_missing} after "
          f">50%-missing drop -> {len(keep)} after constant drop; "
          f"train {len(X_train)} / test {len(X_test)} "
          f"(test fails {int(y_test.sum())})", flush=True)
    preprocessing = {
        "missing_drop_threshold": ">50% missing on train split",
        "n_features_raw": len(feature_cols),
        "n_features_after_missing_drop": n_after_missing,
        "n_features_surviving": len(keep),
        "imputation": "train-split column medians",
        "scaling": "StandardScaler fit on train split",
        "surviving_features": keep,
    }

    clf_models = {
        "logistic_regression": (
            LogisticRegression(max_iter=1000, class_weight="balanced",
                               random_state=SEED),
            {"max_iter": 1000, "class_weight": "balanced"}),
        "random_forest": (
            RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                   random_state=SEED, n_jobs=-1),
            {"n_estimators": 300, "class_weight": "balanced", "n_jobs": -1}),
    }
    for name, (est, params) in clf_models.items():
        rec = base_record("secom", "A", name, params)
        rec["split"] = {
            "protocol": "stratified 80/20, seed 42",
            "n_train": int(len(X_train)), "n_test": int(len(X_test)),
            "train_fails": int(y_train.sum()), "test_fails": int(y_test.sum()),
            "all_negative_baseline_accuracy": float((y_test == 0).mean()),
        }
        rec["preprocessing"] = preprocessing
        rec["notes"] = ("Only 104 fails in the full dataset (~21 in test), so "
                        "recall/F1 estimates are high-variance. Features are "
                        "anonymised; no physical interpretation is possible.")
        t0 = time.perf_counter()
        est.fit(X_train_s, y_train)
        rec["timing"]["train_s"] = round(time.perf_counter() - t0, 4)
        y_pred, ms = timed_predict(est.predict, X_test_s)
        rec["timing"]["infer_ms_per_row"] = round(ms, 5)
        y_score = est.predict_proba(X_test_s)[:, 1]
        rec["metrics"] = classification_metrics(y_test, y_pred, y_score)
        save_record(rec, {"f1": round(rec["metrics"]["f1"], 4),
                          "recall": round(rec["metrics"]["recall"], 4),
                          "roc_auc": round(rec["metrics"]["roc_auc"], 4)})

    # Task C — IsolationForest fitted on PASS rows of the train split only;
    # contamination set to the observed full-dataset fail rate (a prior,
    # stated here, not fitted from test features).
    contamination = float(y.mean())
    iso = IsolationForest(n_estimators=200, contamination=contamination,
                          random_state=SEED, n_jobs=-1)
    rec = base_record("secom", "C", "isolation_forest",
                      {"n_estimators": 200,
                       "contamination": round(contamination, 4),
                       "fit_subset": "pass rows of train split only"})
    rec["split"] = {
        "protocol": "stratified 80/20, seed 42 (same split as Task A)",
        "n_train": int(len(X_train)), "n_test": int(len(X_test)),
        "n_fit_rows_pass_only": int((y_train == 0).sum()),
        "test_fails": int(y_test.sum()),
    }
    rec["preprocessing"] = preprocessing
    rec["notes"] = ("Anomaly view: model never sees fail labels in fitting; "
                    "labels are used only to evaluate. Contamination = "
                    "observed dataset fail rate 104/1567.")
    t0 = time.perf_counter()
    iso.fit(X_train_s[y_train == 0])
    rec["timing"]["train_s"] = round(time.perf_counter() - t0, 4)
    raw_pred, ms = timed_predict(iso.predict, X_test_s)
    rec["timing"]["infer_ms_per_row"] = round(ms, 5)
    y_pred = (raw_pred == -1).astype(int)
    m = classification_metrics(y_test, y_pred)
    tn, fp = m["confusion_matrix"][0]
    m["false_alarm_rate"] = float(fp / (fp + tn)) if (fp + tn) else 0.0
    rec["metrics"] = m
    save_record(rec, {"f1": round(m["f1"], 4), "recall": round(m["recall"], 4),
                      "false_alarm_rate": round(m["false_alarm_rate"], 4)})


# -------------------------------------------------------------------- FD003
def _fd003_data():
    from platform_lib.adapters.cmapss import CMapssAdapter
    from platform_lib.rul_pipeline import build_rul_data

    adapter = CMapssAdapter(RAW / "cmapss" / "extracted", subset="FD003")
    train, meta = adapter.load()
    test, rul = adapter.load_test()
    data = build_rul_data(train, test, rul, cap=125, seq_len=30, seed=SEED)
    return meta, data


def _fd003_record(model: str, params: dict, meta: dict, data) -> dict:
    rec = base_record("cmapss_fd003", "B", model, params)
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
    rec["notes"] = ("FD003 = 1 operating condition, 2 fault modes (HPC + fan). "
                    "All 3 settings fell below the near-constant threshold and "
                    "were dropped by the data-derived rule; sensor_6 was KEPT "
                    "(unlike FD001) because its FD003 std is ~0.018. "
                    + meta.get("data_nature", ""))
    rec["extra"] = {"candidate_train_stds": data.candidate_stds}
    return rec


def run_fd003_ridge() -> None:
    from sklearn.linear_model import Ridge

    from predicai_ml.evaluate import evaluate_all

    print("=== C-MAPSS FD003 — Task B — Ridge (flattened windows) ===",
          flush=True)
    meta, data = _fd003_data()
    print(f"  features ({len(data.features)}): {data.features}", flush=True)
    print(f"  dropped: {[d['column'] for d in data.dropped]}", flush=True)
    Xtr = data.X_train.reshape(len(data.X_train), -1)
    Xva = data.X_val.reshape(len(data.X_val), -1)
    Xte = data.X_test_last.reshape(len(data.X_test_last), -1)

    rec = _fd003_record("ridge_flat", {"alpha": 1.0,
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


def run_fd003_lstm() -> None:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    from predicai_ml.evaluate import evaluate_all
    from predicai_ml.model import LSTMRULRegressor, predict_rul

    print("=== C-MAPSS FD003 — Task B — LSTM (local CPU) ===", flush=True)
    meta, data = _fd003_data()
    cfg = dict(hidden_1=128, hidden_2=64, head_dim=32, dropout=0.2,
               epochs=60, batch_size=64, lr=1e-3, weight_decay=1e-4,
               grad_clip=1.0, patience=10, seed=SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    rec = _fd003_record("lstm", {**cfg, "device": "cpu (local run)",
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
        print(f"  fd003-lstm epoch {epoch:02d}  val RMSE {val_rmse:.3f}  "
              f"({(time.time() - t0) / 60:.1f} min)", flush=True)
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


# ----------------------------------------------------------------------- CLI
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", choices=["ai4i", "secom", "fd003", "all"],
                        required=True)
    args = parser.parse_args()
    if args.set in ("ai4i", "all"):
        run_ai4i()
    if args.set in ("secom", "all"):
        run_secom()
    if args.set in ("fd003", "all"):
        run_fd003_ridge()
        run_fd003_lstm()
    print("done.", flush=True)


if __name__ == "__main__":
    main()
