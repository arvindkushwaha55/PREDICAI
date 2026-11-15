"""PREDICAI platform — benchmarking wave 4 (model expansion + leakage-safe
evaluation, 2026-11-15).

Wave 4 broadens the model zoo where waves 1-2 used one model family per
task, and — the point of the wave — replaces the two weakest split
protocols in the earlier records with leakage-safe ones:

- FD001 GRU: same shipped data/protocol/recipe as the FD001 LSTM, GRU
  cells instead of LSTM cells (architecture comparison, in-domain).
- FD003 CNN-LSTM: convolutional frontend + the shipped LSTM stack; the
  trained checkpoint is SAVED (platform/artifacts/) for the wave-5
  FD003 -> FD001 transfer, which wave 2 had to skip for lack of one.
- FD002 tree regressors (RF + XGBoost) on flattened windows.
- AI4I SVM (RBF) + MLP, same stratified split as wave 1.
- SECOM Task C: One-Class SVM + MLP autoencoder, both fitted on PASS
  rows only (same discipline as the wave-1 Isolation Forest).
- CWRU RECORDING-WISE re-evaluation: 12 official recordings (3 per
  class: normal at 0/1/2 hp; faults 0.007/0.014/0.021 inch at 0 hp),
  leave-one-recording-out — no window of a test recording is ever in
  training. This is the honest counterpart of wave 2's window-split
  1.000, and the two numbers are reported side by side, never merged.
- NASA Battery RUL: leave-one-battery-out over B0005/B0006/B0018
  (B0007 excluded — censored, never crosses the 1.4 Ah EOL threshold).
- Paderborn Task A: Random Forest with a MEASUREMENT-GROUPED split
  (GroupShuffleSplit by measurement file) — still one physical bearing
  per class, so it measures within-bearing separability across
  measurements, not cross-bearing generalisation.

Usage (from the platform/ directory):
    ../.venv/bin/python -m platform_lib.benchmarking.run_wave4 --set gru
    ... --set cnnlstm | fd002trees | ai4i | secom | cwru | battery |
        paderborn | all

Seed is 42 everywhere, matching waves 1-2.
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
from platform_lib.benchmarking.run_wave2 import (  # noqa: E402
    LSTM_CFG,
    _subset_data,
    _subset_record,
)

ARTIFACTS_DIR = PLATFORM_DIR / "artifacts"


# ------------------------------------------------- shared torch training
def _train_seq_model(model, X_train, y_train, X_val, y_val, cfg: dict):
    """The shipped recipe (AdamW + cosine + MSE + grad-clip + early stop
    on val RMSE with best-state restore), shared by the wave-4 sequence
    models. Returns (train_s, epochs_run, best_val_rmse)."""
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    from predicai_ml.model import predict_rul

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    dl = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True)
    x_va = torch.from_numpy(X_val)
    y_va = torch.from_numpy(y_val)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                            weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg["epochs"])
    loss_fn = nn.MSELoss()
    best_rmse, bad_epochs, epochs_run = float("inf"), 0, 0
    best_state = None
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
        print(f"    epoch {epoch:02d}  val RMSE {val_rmse:.3f}  "
              f"({(time.time() - t0) / 60:.1f} min)", flush=True)
        if val_rmse < best_rmse - 1e-4:
            best_rmse, bad_epochs = val_rmse, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= cfg["patience"]:
                print(f"    early stop at epoch {epoch}", flush=True)
                break
    model.load_state_dict(best_state)
    return time.time() - t0, epochs_run, best_rmse


def _seq_metrics(model, data):
    import torch

    from predicai_ml.evaluate import evaluate_all
    from predicai_ml.model import predict_rul

    x_te = torch.from_numpy(data.X_test_last)
    x_va = torch.from_numpy(data.X_val)
    with torch.no_grad():
        pred_te_t, ms = timed_predict(
            lambda X: predict_rul(model, X), x_te, repeats=50)
        pred_va = predict_rul(model, x_va).numpy()
    metrics = {
        "test_last_window_per_engine": evaluate_all(
            data.y_test_last, pred_te_t.numpy()),
        "val_all_windows": evaluate_all(data.y_val, pred_va),
    }
    return metrics, round(ms, 5)


# ---------------------------------------------------------- 1) FD001 GRU
def run_fd001_gru() -> None:
    import torch
    from torch import nn

    from predicai_ml.preprocessing import FEATURES as FD001_FEATURES
    from predicai_ml.preprocessing import run_pipeline
    from platform_lib.adapters.cmapss import CMapssAdapter

    class GRURULRegressor(nn.Module):
        """Mirror of LSTMRULRegressor with GRU cells (same widths/head)."""

        def __init__(self, num_features, hidden_1=128, hidden_2=64,
                     head_dim=32, dropout=0.2):
            super().__init__()
            self.gru1 = nn.GRU(num_features, hidden_1, batch_first=True)
            self.drop1 = nn.Dropout(dropout)
            self.gru2 = nn.GRU(hidden_1, hidden_2, batch_first=True)
            self.drop2 = nn.Dropout(dropout)
            self.head = nn.Sequential(nn.Linear(hidden_2, head_dim),
                                      nn.ReLU(), nn.Linear(head_dim, 1))

        def forward(self, x):
            out, _ = self.gru1(x)
            out = self.drop1(out)
            out, _ = self.gru2(out)
            last = self.drop2(out[:, -1, :])
            return self.head(last).squeeze(-1)

    print("=== C-MAPSS FD001 — Task B — GRU (local CPU) ===", flush=True)
    adapter = CMapssAdapter(RAW / "cmapss" / "extracted", subset="FD001")
    train, meta = adapter.load()
    test, rul = adapter.load_test()
    # The SHIPPED FD001 pipeline (predicai_ml.preprocessing.run_pipeline):
    # the exact data the shipped LSTM was trained/evaluated on.
    data = run_pipeline(train, test, rul, cap=125, seq_len=30)
    print(f"  features ({len(FD001_FEATURES)}): {FD001_FEATURES}", flush=True)
    print(f"  train windows {len(data.X_train)} / val {len(data.X_val)}",
          flush=True)

    cfg = dict(LSTM_CFG)
    rec = base_record(
        "cmapss_fd001", "B", "gru",
        {**cfg, "device": "cpu (local run)",
         "architecture": "GRU(128)->GRU(64)->head 32->1 (mirror of the "
                         "shipped LSTM with GRU cells)"})
    rec["split"] = {
        "protocol": "engine-wise 80/20 train/val (seed 42) + official "
                    "provided test set, last window per engine — identical "
                    "to the shipped FD001 LSTM evaluation",
        "n_train_engines": len(data.train_engines),
        "n_val_engines": len(data.val_engines),
        "n_train_windows": int(len(data.X_train)),
        "n_val_windows": int(len(data.X_val)),
        "n_test_engines": int(len(data.X_test_last)),
    }
    rec["preprocessing"] = {
        "features": FD001_FEATURES,
        "n_features": len(FD001_FEATURES),
        "feature_derivation": "shipped FD001 pipeline "
                              "(predicai_ml.preprocessing): 14 sensors; "
                              "6 constant sensors + sensor_6 dropped, "
                              "settings not used (single condition)",
        "rul_cap": 125, "seq_len": 30,
        "scaling": "z-score (ddof=0) fit on train engines only",
    }
    rec["notes"] = ("Architecture comparison on identical data/protocol: "
                    "GRU vs the shipped LSTM (owner Kaggle GPU run, test "
                    "RMSE 13.73). " + meta.get("data_nature", ""))
    model = GRURULRegressor(len(FD001_FEATURES), cfg["hidden_1"],
                            cfg["hidden_2"], cfg["head_dim"], cfg["dropout"])
    train_s, epochs_run, best_val = _train_seq_model(
        model, data.X_train, data.y_train, data.X_val, data.y_val, cfg)
    rec["timing"]["train_s"] = round(train_s, 1)
    rec["extra"]["epochs_run"] = epochs_run
    rec["extra"]["best_val_rmse"] = round(best_val, 4)
    metrics, ms = _seq_metrics(model, data)
    rec["timing"]["infer_ms_per_row"] = ms
    rec["metrics"] = metrics
    shipped = json.loads((REPO_ROOT / "ml" / "artifacts" /
                          "metrics.json").read_text())[
        "test_last_window_per_engine"]
    rec["extra"]["reference_shipped_lstm_fd001"] = {
        "rmse": shipped["rmse"], "mae": shipped["mae"],
        "nasa_score": shipped["nasa_score"],
        "source": "ml/artifacts/metrics.json (shipped model, same "
                  "last-window protocol)"}
    rec["extra"]["delta_vs_shipped_lstm"] = {
        "rmse_delta": metrics["test_last_window_per_engine"]["rmse"]
                      - shipped["rmse"],
        "mae_delta": metrics["test_last_window_per_engine"]["mae"]
                     - shipped["mae"],
        "nasa_delta": metrics["test_last_window_per_engine"]["nasa_score"]
                      - shipped["nasa_score"]}
    save_record(rec, {"test_rmse": round(
        metrics["test_last_window_per_engine"]["rmse"], 3),
        "test_mae": round(metrics["test_last_window_per_engine"]["mae"], 3),
        "test_nasa": round(
            metrics["test_last_window_per_engine"]["nasa_score"], 1)})


# ----------------------------------------------------- 2) FD003 CNN-LSTM
def run_fd003_cnn_lstm() -> None:
    import torch
    from torch import nn

    class CNNLSTMRegressor(nn.Module):
        """Conv1d frontend -> the shipped 2-layer LSTM stack + head."""

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

    print("=== C-MAPSS FD003 — Task B — CNN-LSTM (local CPU) ===", flush=True)
    meta, data = _subset_data("FD003")
    print(f"  features ({len(data.features)}): {data.features}", flush=True)
    cfg = dict(LSTM_CFG)
    params = {**cfg, "conv_1": 64, "conv_2": 32, "kernel_size": 3,
              "device": "cpu (local run)",
              "architecture": "Conv1d(F->64,k3)->ReLU->Conv1d(64->32,k3)"
                              "->ReLU->LSTM(128)->LSTM(64)->head 32->1 "
                              "(F = FD003's 15 derived features)"}
    rec = _subset_record("FD003", "cnn_lstm", params, meta, data)
    model = CNNLSTMRegressor(len(data.features), 64, 32, cfg["hidden_1"],
                             cfg["hidden_2"], cfg["head_dim"],
                             cfg["dropout"])
    train_s, epochs_run, best_val = _train_seq_model(
        model, data.X_train, data.y_train, data.X_val, data.y_val, cfg)
    rec["timing"]["train_s"] = round(train_s, 1)
    rec["extra"]["epochs_run"] = epochs_run
    rec["extra"]["best_val_rmse"] = round(best_val, 4)

    # Save the checkpoint + a preprocessing sidecar (features + scaler,
    # same contract shape as ml/artifacts/preprocessing.json) so wave 5
    # can run the FD003 -> FD001 transfer without refitting anything.
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = ARTIFACTS_DIR / "fd003_cnn_lstm.pt"
    torch.save(model.state_dict(), ckpt_path)
    sidecar = {
        "dataset": "cmapss_fd003",
        "model": "cnn_lstm",
        "features": data.features,
        "num_features": len(data.features),
        "seq_len": 30,
        "rul_cap": 125,
        "scaler": {"type": "zscore", "ddof": 0,
                   "mean": [float(v) for v in data.scaler.mean],
                   "scale": [float(v) for v in data.scaler.scale]},
        "architecture": {"conv_1": 64, "conv_2": 32, "kernel_size": 3,
                         "hidden_1": cfg["hidden_1"],
                         "hidden_2": cfg["hidden_2"],
                         "head_dim": cfg["head_dim"],
                         "dropout": cfg["dropout"]},
        "train_engines": data.train_engines,
        "val_engines": data.val_engines,
    }
    sidecar_path = ARTIFACTS_DIR / "fd003_cnn_lstm_preprocessing.json"
    sidecar_path.write_text(json.dumps(sidecar, indent=2))
    rec["extra"]["checkpoint"] = {
        "state_dict": "platform/artifacts/fd003_cnn_lstm.pt",
        "preprocessing": "platform/artifacts/"
                         "fd003_cnn_lstm_preprocessing.json",
        "purpose": "saved for the wave-5 FD003 -> FD001 zero-shot "
                   "transfer (wave 2 had to skip the reverse direction "
                   "because the wave-1 FD003 LSTM run saved no "
                   "checkpoint)",
    }
    metrics, ms = _seq_metrics(model, data)
    rec["timing"]["infer_ms_per_row"] = ms
    rec["metrics"] = metrics
    rec["notes"] += (" Checkpoint saved to platform/artifacts/"
                     "fd003_cnn_lstm.pt (best-val state) for wave 5.")
    save_record(rec, {"test_rmse": round(
        metrics["test_last_window_per_engine"]["rmse"], 3),
        "test_mae": round(metrics["test_last_window_per_engine"]["mae"], 3),
        "test_nasa": round(
            metrics["test_last_window_per_engine"]["nasa_score"], 1)})


# ------------------------------------------------- 3) FD002 tree regressors
def run_fd002_trees() -> None:
    from sklearn.ensemble import RandomForestRegressor

    from predicai_ml.evaluate import evaluate_all

    print("=== C-MAPSS FD002 — Task B — tree regressors (flattened) ===",
          flush=True)
    meta, data = _subset_data("FD002")
    Xtr = data.X_train.reshape(len(data.X_train), -1)
    Xva = data.X_val.reshape(len(data.X_val), -1)
    Xte = data.X_test_last.reshape(len(data.X_test_last), -1)
    max_train = 10_000
    if len(Xtr) > max_train:
        rng = np.random.default_rng(SEED)
        sub = np.sort(rng.choice(len(Xtr), size=max_train, replace=False))
        Xtr_fit, ytr_fit = Xtr[sub], data.y_train[sub]
        subsample_note = (f"training windows subsampled to {max_train} of "
                          f"{len(data.X_train)} (uniform, seed 42) for "
                          f"CPU feasibility; val/test are NOT subsampled")
    else:
        Xtr_fit, ytr_fit = Xtr, data.y_train
        subsample_note = "no subsampling"
    print(f"  flat dim {Xtr.shape[1]}; fit rows {len(Xtr_fit)} "
          f"({subsample_note})", flush=True)

    import xgboost as xgb

    models = {
        "random_forest_reg": (
            RandomForestRegressor(n_estimators=100, max_features=0.3,
                                  random_state=SEED, n_jobs=-1),
            {"n_estimators": 100, "n_jobs": -1,
             "max_features": 0.3,
             "cpu_feasibility_note": "config reduced for this VM's CPU "
                                  "after three attempts did not finish "
                                  "the RF fit stage in ~15 min each: "
                                  "300 trees at max_features=1.0 "
                                  "(subsample 20k), 300 trees at "
                                  "max_features=1.0 (subsample 10k), "
                                  "300 trees at max_features=0.3 "
                                  "(subsample 10k). Final: 100 trees, "
                                  "max_features=0.3 (the classic RF "
                                  "decorrelation setting), subsample "
                                  "10k. The recorded train_s is this "
                                  "final run's.",
             "input": "flattened (30 x 23) windows, predictions clipped "
                      "at >= 0", "subsample": subsample_note}),
        "xgboost_reg": (
            xgb.XGBRegressor(n_estimators=500, max_depth=6,
                             learning_rate=0.05, subsample=0.9,
                             colsample_bytree=0.8, random_state=SEED,
                             n_jobs=-1),
            {"n_estimators": 500, "max_depth": 6, "learning_rate": 0.05,
             "subsample": 0.9, "colsample_bytree": 0.8,
             "input": "flattened (30 x 23) windows, predictions clipped "
                      "at >= 0", "training_subsample": subsample_note}),
    }
    for name, (est, params) in models.items():
        rec = _subset_record("FD002", name, params, meta, data)
        rec["extra"]["training_windows_used"] = int(len(Xtr_fit))
        t0 = time.perf_counter()
        est.fit(Xtr_fit, ytr_fit)
        rec["timing"]["train_s"] = round(time.perf_counter() - t0, 4)
        pred_te, ms = timed_predict(
            lambda X: np.clip(est.predict(X), 0, None), Xte, repeats=20)
        rec["timing"]["infer_ms_per_row"] = round(ms, 5)
        pred_va = np.clip(est.predict(Xva), 0, None)
        rec["metrics"] = {
            "test_last_window_per_engine": evaluate_all(
                data.y_test_last, pred_te),
            "val_all_windows": evaluate_all(data.y_val, pred_va),
        }
        save_record(rec, {"test_rmse": round(
            rec["metrics"]["test_last_window_per_engine"]["rmse"], 3),
            "test_mae": round(
                rec["metrics"]["test_last_window_per_engine"]["mae"], 3),
            "test_nasa": round(
                rec["metrics"]["test_last_window_per_engine"]
                ["nasa_score"], 1)})


# ------------------------------------------------------------ 4) AI4I SVM/MLP
def _ai4i_prepared():
    from sklearn.model_selection import train_test_split

    from platform_lib.adapters.ai4i import AI4IAdapter

    df, meta = AI4IAdapter(RAW / "ai4i").load()
    X = df[meta["numeric_features"]].copy()
    X = pd.concat([X, pd.get_dummies(df["Type"], prefix="Type",
                                     dtype=float)], axis=1)
    y = df[meta["target"]].astype(int)
    feature_names = list(X.columns)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)
    return meta, feature_names, X_train, X_test, y_train, y_test


def run_ai4i_svm_mlp() -> None:
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    print("=== AI4I 2020 — Task A — SVM (RBF) + MLP ===", flush=True)
    meta, feature_names, X_train, X_test, y_train, y_test = _ai4i_prepared()
    baseline_acc = float((y_test == 0).mean())
    models = {
        "svm": (
            Pipeline([("scaler", StandardScaler()),
                      ("clf", SVC(kernel="rbf", C=1.0, gamma="scale",
                                  class_weight="balanced",
                                  probability=True, random_state=SEED))]),
            {"pipeline": "StandardScaler + SVC(kernel='rbf', C=1.0, "
                         "gamma='scale')",
             "class_weight": "balanced",
             "probability": "Platt scaling (internal 5-fold) for "
                            "ROC-AUC scores"}),
        "mlp": (
            Pipeline([("scaler", StandardScaler()),
                      ("clf", MLPClassifier(hidden_layer_sizes=(64, 32),
                                            activation="relu",
                                            solver="adam", alpha=0.0001,
                                            max_iter=500,
                                            random_state=SEED))]),
            {"pipeline": "StandardScaler + MLPClassifier",
             "hidden_layer_sizes": [64, 32], "activation": "relu",
             "solver": "adam", "alpha": 0.0001, "max_iter": 500}),
    }
    for name, (est, params) in models.items():
        rec = base_record("ai4i2020", "A", name, params)
        rec["split"] = {
            "protocol": "stratified 80/20, seed 42 (identical to wave-1 "
                        "AI4I records)",
            "n_train": int(len(X_train)), "n_test": int(len(X_test)),
            "train_failures": int(y_train.sum()),
            "test_failures": int(y_test.sum()),
            "all_negative_baseline_accuracy": baseline_acc,
        }
        rec["preprocessing"] = {
            "features": feature_names,
            "excluded_id_columns": meta["id_columns"],
            "excluded_leakage_columns": meta["leakage_columns"],
            "categorical_encoding": "one-hot of Type (fixed domain "
                                    "levels L/M/H)",
            "scaling": "StandardScaler inside pipeline (fit on train "
                       "split only)",
        }
        rec["notes"] = ("Synthetic dataset; failure modes are "
                        "deterministic rules of the features, so high "
                        "scores reflect the rule structure, not "
                        "real-world machine performance. Failure-mode "
                        "flag columns excluded (target leakage).")
        t0 = time.perf_counter()
        est.fit(X_train, y_train)
        rec["timing"]["train_s"] = round(time.perf_counter() - t0, 4)
        y_pred, ms = timed_predict(est.predict, X_test)
        rec["timing"]["infer_ms_per_row"] = round(ms, 5)
        y_score = est.predict_proba(X_test)[:, 1]
        rec["metrics"] = classification_metrics(y_test, y_pred, y_score)
        if name == "mlp":
            rec["extra"]["n_iter_actual"] = int(
                est.named_steps["clf"].n_iter_)
        save_record(rec, {"f1": round(rec["metrics"]["f1"], 4),
                          "roc_auc": round(rec["metrics"]["roc_auc"], 4),
                          "recall": round(rec["metrics"]["recall"], 4)})


# ------------------------------------------------- 5) SECOM Task C: OCSVM + AE
def _secom_prepared():
    """Wave-1 SECOM preprocessing, replicated exactly (same seed/split):
    train-fitted >50%-missing drop, constant drop, median imputation,
    StandardScaler. Returns scaled frames + audit info."""
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    from platform_lib.adapters.secom import SECOMAdapter

    df, meta = SECOMAdapter(RAW / "secom").load()
    feature_cols = meta["feature_columns"]
    y = df["target"]
    idx_train, idx_test = train_test_split(
        df.index, test_size=0.2, stratify=y, random_state=SEED)
    train_df, test_df = df.loc[idx_train], df.loc[idx_test]
    y_train, y_test = y.loc[idx_train], y.loc[idx_test]
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
    preprocessing = {
        "missing_drop_threshold": ">50% missing on train split",
        "n_features_raw": len(feature_cols),
        "n_features_after_missing_drop": n_after_missing,
        "n_features_surviving": len(keep),
        "imputation": "train-split column medians",
        "scaling": "StandardScaler fit on train split",
        "surviving_features": keep,
    }
    return y, y_train, y_test, X_train_s, X_test_s, preprocessing


def _anomaly_record(model: str, params: dict, y_train, y_test,
                    preprocessing, n_fit: int) -> dict:
    rec = base_record("secom", "C", model, params)
    rec["split"] = {
        "protocol": "stratified 80/20, seed 42 (same split as wave-1 "
                    "SECOM records); model fitted on PASS rows of the "
                    "train split only — fail labels used only to "
                    "evaluate",
        "n_train": int(len(y_train)), "n_test": int(len(y_test)),
        "n_fit_rows_pass_only": int(n_fit),
        "test_fails": int(y_test.sum()),
    }
    rec["preprocessing"] = preprocessing
    rec["notes"] = ("Anomaly view: the model never sees fail labels in "
                    "fitting. Only 104 fails exist in the full dataset "
                    "(~21 in test), so recall/F1 estimates are "
                    "high-variance. Features are anonymised; no physical "
                    "interpretation is possible.")
    return rec


def _finish_anomaly_record(rec, y_test, y_pred, headline_extra=None):
    m = classification_metrics(y_test, y_pred)
    tn, fp = m["confusion_matrix"][0]
    m["false_alarm_rate"] = float(fp / (fp + tn)) if (fp + tn) else 0.0
    rec["metrics"] = m
    headline = {"f1": round(m["f1"], 4), "recall": round(m["recall"], 4),
                "false_alarm_rate": round(m["false_alarm_rate"], 4)}
    if headline_extra:
        headline.update(headline_extra)
    save_record(rec, headline)


def run_secom_anomaly() -> None:
    from sklearn.svm import OneClassSVM

    print("=== SECOM — Task C — One-Class SVM + Autoencoder ===", flush=True)
    y_all, y_train, y_test, X_train_s, X_test_s, preprocessing = \
        _secom_prepared()
    fail_rate = float(y_all.mean())  # 104/1567 — a prior, not fitted
    X_pass = X_train_s[y_train == 0]
    print(f"  surviving features: {preprocessing['n_features_surviving']}; "
          f"pass fit rows: {len(X_pass)}; fail-rate prior: {fail_rate:.4f}",
          flush=True)

    # --- One-Class SVM ---
    ocsvm = OneClassSVM(kernel="rbf", gamma="scale", nu=fail_rate)
    rec = _anomaly_record(
        "one_class_svm",
        {"kernel": "rbf", "gamma": "scale", "nu": round(fail_rate, 4),
         "nu_rationale": "nu set to the observed full-dataset fail rate "
                         "(104/1567) — an upper bound on the outlier "
                         "fraction, stated as a prior",
         "fit_subset": "pass rows of train split only"},
        y_train, y_test, preprocessing, len(X_pass))
    t0 = time.perf_counter()
    ocsvm.fit(X_pass)
    rec["timing"]["train_s"] = round(time.perf_counter() - t0, 4)
    raw_pred, ms = timed_predict(ocsvm.predict, X_test_s)
    rec["timing"]["infer_ms_per_row"] = round(ms, 5)
    _finish_anomaly_record(rec, y_test, (raw_pred == -1).astype(int))

    # --- MLP autoencoder ---
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    n_feat = X_pass.shape[1]

    class AE(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.enc = nn.Sequential(nn.Linear(d, 64), nn.ReLU(),
                                     nn.Linear(64, 16))
            self.dec = nn.Sequential(nn.Linear(16, 64), nn.ReLU(),
                                     nn.Linear(64, d))

        def forward(self, x):
            return self.dec(self.enc(x))

    ae = AE(n_feat)
    opt = torch.optim.Adam(ae.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    x_pass_t = torch.tensor(X_pass.to_numpy(), dtype=torch.float32)
    dl = DataLoader(TensorDataset(x_pass_t), batch_size=256, shuffle=True,
                    generator=torch.Generator().manual_seed(SEED))
    epochs = 30
    t0 = time.perf_counter()
    for _ in range(epochs):
        ae.train()
        for (xb,) in dl:
            opt.zero_grad()
            loss = loss_fn(ae(xb), xb)
            loss.backward()
            opt.step()
    train_s = time.perf_counter() - t0

    def recon_errors(X: pd.DataFrame) -> np.ndarray:
        ae.eval()
        with torch.no_grad():
            xt = torch.tensor(X.to_numpy(), dtype=torch.float32)
            rec_x = ae(xt)
            return ((rec_x - xt) ** 2).mean(dim=1).numpy()

    train_scores = recon_errors(X_pass)
    threshold = float(np.quantile(train_scores, 1.0 - fail_rate))
    rec = _anomaly_record(
        "autoencoder",
        {"architecture": f"MLP autoencoder {n_feat}->64->16->64->"
                         f"{n_feat} (ReLU hidden, linear output)",
         "epochs": epochs, "batch_size": 256, "lr": 0.001,
         "optimizer": "Adam", "loss": "MSE reconstruction error",
         "fit_subset": "pass rows of train split only",
         "threshold_rule": "anomaly score above the (1 - fail_rate) "
                           "quantile of TRAIN pass reconstruction "
                           "errors, fail_rate = 104/1567 (prior)"},
        y_train, y_test, preprocessing, len(X_pass))
    rec["timing"]["train_s"] = round(train_s, 4)
    test_scores, ms = timed_predict(recon_errors, X_test_s, repeats=10)
    rec["timing"]["infer_ms_per_row"] = round(ms, 5)
    rec["extra"]["threshold"] = threshold
    rec["extra"]["train_score_quantiles"] = {
        "p50": float(np.quantile(train_scores, 0.5)),
        "p90": float(np.quantile(train_scores, 0.9)),
        "p99": float(np.quantile(train_scores, 0.99))}
    _finish_anomaly_record(rec, y_test,
                           (test_scores > threshold).astype(int))


# ------------------------------------------- 6) CWRU recording-wise (LOGO)
# Wave-4 recording catalogue: the wave-2 four (0.007", 0 hp) plus the
# official 0.014" / 0.021" files at 0 hp and two more normal baselines
# (1 hp, 2 hp) — 12 recordings, 3 per class, all 12 kHz drive-end.
CWRU_W4_RECORDINGS = {
    "97.mat": ("normal_0", "Normal"),
    "98.mat": ("normal_1", "Normal"),
    "99.mat": ("normal_2", "Normal"),
    "105.mat": ("IR007_0", "InnerRace"),
    "169.mat": ("IR014_0", "InnerRace"),
    "209.mat": ("IR021_0", "InnerRace"),
    "118.mat": ("B007_0", "Ball"),
    "185.mat": ("B014_0", "Ball"),
    "222.mat": ("B021_0", "Ball"),
    "130.mat": ("OR007@6_0", "OuterRace"),
    "197.mat": ("OR014@6_0", "OuterRace"),
    "234.mat": ("OR021@6_0", "OuterRace"),
}


def _cwru_w4_frame() -> tuple[pd.DataFrame, list[dict]]:
    from scipy.io import loadmat

    from platform_lib.adapters.cwru import (FEATURE_COLUMNS, WINDOW_LEN,
                                            window_features)

    rows, catalogue = [], []
    for fname, (condition, label) in CWRU_W4_RECORDINGS.items():
        path = RAW / "cwru" / fname
        if not path.exists():
            raise FileNotFoundError(f"CWRU recording missing: {path}")
        mat = loadmat(path)
        # Select the variable matching THIS file's id (99.mat as served
        # by the official host bundles X098 alongside X099; X099 is the
        # self-identified 2 hp normal recording and is the one used).
        key = f"X{int(fname.split('.')[0]):03d}_DE_time"
        if key not in mat:
            raise ValueError(f"{fname}: expected variable {key}, found "
                             f"{[k for k in mat if k.endswith('_DE_time')]}")
        series = np.asarray(mat[key], dtype=np.float64).ravel()
        n_windows = len(series) // WINDOW_LEN
        for i in range(n_windows):
            feats = window_features(
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


def run_cwru_recordingwise() -> None:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (accuracy_score, classification_report,
                                 confusion_matrix, f1_score)

    from platform_lib.adapters.cwru import FEATURE_COLUMNS, LABELS

    print("=== CWRU — Task A — Random Forest, RECORDING-WISE (LOGO) ===",
          flush=True)
    df, catalogue = _cwru_w4_frame()
    print(f"  windows: {len(df)} over {df['recording'].nunique()} "
          f"recordings; per class: {df['label'].value_counts().to_dict()}",
          flush=True)
    X = df[FEATURE_COLUMNS]
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
    metrics = {
        "accuracy": float(accuracy_score(y, y_pred)),
        "f1_macro": float(f1_score(y, y_pred, average="macro",
                                   zero_division=0)),
        "confusion_matrix": confusion_matrix(
            y, y_pred, labels=labels).tolist(),
        "confusion_labels": labels,
        "per_class": {c: {"precision": report[c]["precision"],
                          "recall": report[c]["recall"],
                          "f1": report[c]["f1-score"],
                          "support": int(report[c]["support"])}
                      for c in labels},
    }
    wave2 = json.loads((PLATFORM_DIR / "experiments" /
                        "cwru__taskA__random_forest__s42.json").read_text())
    rec = base_record("cwru", "A", "random_forest_recordingwise",
                      {"n_estimators": 300, "class_weight": "balanced",
                       "n_jobs": -1,
                       "features": "per-window time-domain statistics of "
                                   "the drive-end vibration series "
                                   "(identical features to wave 2)"})
    rec["experiment_id"] = "cwru__taskA__random_forest_recordingwise__s42"
    rec["split"] = {
        "protocol": "RECORDING-WISE leave-one-recording-out "
                    "(LeaveOneGroupOut over 12 recordings; groups = "
                    "recording id): in every fold the held-out "
                    "recording's windows are the test set and NO window "
                    "from that recording appears in training. Metrics "
                    "are computed on the pooled out-of-fold predictions.",
        "n_recordings": int(df["recording"].nunique()),
        "n_windows": int(len(df)),
        "recordings_per_class": {k: int(v) for k, v in
                                 y.groupby(y).count().items()},
        "class_window_counts": {k: int(v) for k, v in
                                y.value_counts().items()},
    }
    rec["preprocessing"] = {
        "features": FEATURE_COLUMNS,
        "windowing": "non-overlapping windows of 2048 samples over the "
                     "DE channel of each recording (identical to wave 2)",
        "recordings": catalogue,
    }
    rec["timing"] = {"train_s": round(total_train_s, 4),
                     "infer_ms_per_row": round(
                         total_pred_s / len(df) * 1000.0, 5)}
    rec["metrics"] = metrics
    rec["notes"] = (
        "The leakage-safe counterpart of the wave-2 CWRU record. Wave 2 "
        "split WINDOWS of 4 recordings (one per class) stratified 80/20, "
        "so every test window's recording was also in training — its "
        "accuracy/F1 of 1.000 measures within-recording separability "
        "only. Here, 12 recordings (3 per class: Normal at 0/1/2 hp; "
        "InnerRace/Ball/OuterRace at 0.007/0.014/0.021 inch, 0 hp) are "
        "evaluated leave-one-recording-out, so the score below is the "
        "first CWRU number in this project that speaks about unseen "
        "recordings. It still does NOT cover unseen fault diameters "
        "combined with unseen loads jointly, nor different machines. "
        "Data note: 99.mat as served by the official host bundles the "
        "X098 variable alongside X099; the self-identified X099_DE_time "
        "series (2 hp normal) is the one used.")
    rec["extra"] = {
        "per_recording_accuracy": per_recording,
        "wave2_window_split_reference": {
            "accuracy": wave2["metrics"]["accuracy"],
            "f1_macro": wave2["metrics"]["f1_macro"],
            "source": "cwru__taskA__random_forest__s42 (window-wise "
                      "stratified split over 4 recordings — optimistic, "
                      "see its notes)"},
        "delta_vs_wave2_window_split": {
            "accuracy_delta": metrics["accuracy"]
                              - wave2["metrics"]["accuracy"],
            "f1_macro_delta": metrics["f1_macro"]
                              - wave2["metrics"]["f1_macro"]},
    }
    save_record(rec, {"accuracy": round(metrics["accuracy"], 4),
                      "f1_macro": round(metrics["f1_macro"], 4)})


# ------------------------------------------------------ 7) NASA Battery LOBO
BATTERY_LOBO = ["B0005", "B0006", "B0018"]


def _battery_frame():
    from platform_lib.adapters.nasa_battery import (FEATURE_COLUMNS,
                                                    NASABatteryAdapter)

    df, meta = NASABatteryAdapter(RAW / "nasa_battery").load()
    features = ["cycle_index"] + FEATURE_COLUMNS
    return df, meta, features


def _reg_metrics(y_true, y_pred) -> dict:
    from predicai_ml.evaluate import mae, rmse

    return {"rmse": float(rmse(np.asarray(y_true), np.asarray(y_pred))),
            "mae": float(mae(np.asarray(y_true), np.asarray(y_pred)))}


def _battery_record(model: str, params: dict, meta: dict,
                    features: list[str]) -> dict:
    rec = base_record("nasa_battery", "B", model, params)
    rec["experiment_id"] = f"battery__taskB__{model}_lobo__s42"
    rec["split"] = {
        "protocol": "LEAVE-ONE-BATTERY-OUT (LOBO): each of B0005, "
                    "B0006, B0018 is held out in turn and predicted by a "
                    "model fitted on the other two batteries only; no "
                    "cycle of the test battery is in training. B0007 is "
                    "EXCLUDED: it is censored — its capacity never "
                    "crosses the 1.4 Ah end-of-life threshold (last "
                    "recorded 1.432 Ah), so it has no observed EOL "
                    "label.",
        "held_out_batteries": BATTERY_LOBO,
        "excluded_batteries": ["B0007 (censored — never crosses 1.4 Ah "
                               "EOL)"],
    }
    rec["preprocessing"] = {
        "features": features,
        "target": "rul_cycles_eol (discharge cycles until capacity "
                  "first < 1.4 Ah = 70% of the 2 Ah rating)",
        "note": "capacity_ah / soh are measured per-cycle health "
                "indicators and are inputs (standard for capacity-"
                "trajectory RUL); cycle_index counts discharge cycles "
                "of that battery only",
    }
    rec["notes"] = ("Real laboratory battery-aging data (NASA Ames "
                    "PCoE, Saha & Goebel 2007). LOBO over 3 batteries "
                    "is a small-sample evaluation: per-battery numbers "
                    "are reported individually, not hidden in the "
                    "pooled score. " + "")
    return rec


def run_battery_rf() -> None:
    from sklearn.ensemble import RandomForestRegressor

    print("=== NASA Battery — Task B — Random Forest (LOBO) ===",
          flush=True)
    df, meta, features = _battery_frame()
    rec = _battery_record(
        "random_forest",
        {"n_estimators": 300, "n_jobs": -1,
         "features": "cycle_index + per-cycle discharge statistics "
                     "(adapter FEATURE_COLUMNS)"},
        meta, features)
    per_battery, yt_all, yp_all = {}, [], []
    total_train_s, total_pred_s = 0.0, 0.0
    for held in BATTERY_LOBO:
        tr = df[df["battery"].isin([b for b in BATTERY_LOBO if b != held])]
        te = df[df["battery"] == held]
        est = RandomForestRegressor(n_estimators=300, random_state=SEED,
                                    n_jobs=-1)
        t0 = time.perf_counter()
        est.fit(tr[features], tr["rul_cycles_eol"])
        total_train_s += time.perf_counter() - t0
        t0 = time.perf_counter()
        pred = np.clip(est.predict(te[features]), 0, None)
        total_pred_s += time.perf_counter() - t0
        per_battery[held] = {"n_cycles": int(len(te)),
                             **_reg_metrics(te["rul_cycles_eol"], pred)}
        yt_all.append(te["rul_cycles_eol"].to_numpy())
        yp_all.append(pred)
        print(f"    held out {held}: {per_battery[held]}", flush=True)
    rec["metrics"] = {
        "per_battery": per_battery,
        "pooled": _reg_metrics(np.concatenate(yt_all),
                               np.concatenate(yp_all)),
    }
    rec["timing"] = {"train_s": round(total_train_s, 4),
                     "infer_ms_per_row": round(
                         total_pred_s / len(df[df['battery'].isin(
                             BATTERY_LOBO)]) * 1000.0, 5)}
    save_record(rec, {"pooled_rmse": round(rec["metrics"]["pooled"]
                                           ["rmse"], 3),
                      "pooled_mae": round(rec["metrics"]["pooled"]
                                          ["mae"], 3)})


def run_battery_lstm() -> None:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    print("=== NASA Battery — Task B — small LSTM on capacity (LOBO) ===",
          flush=True)
    df, meta, features = _battery_frame()
    window = 10

    class BatteryLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(1, 32, batch_first=True)
            self.head = nn.Sequential(nn.Linear(32, 16), nn.ReLU(),
                                      nn.Linear(16, 1))

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(out[:, -1, :]).squeeze(-1)

    def windows_for(battery_df: pd.DataFrame):
        cap = battery_df["capacity_ah"].to_numpy(dtype=np.float64)
        rul = battery_df["rul_cycles_eol"].to_numpy(dtype=np.float64)
        if len(cap) <= window:
            return (np.empty((0, window, 1), dtype=np.float32),
                    np.empty((0,), dtype=np.float32))
        X = np.stack([cap[i - window:i] for i in range(window, len(cap))])
        y = rul[window:]
        return X[:, :, None].astype(np.float32), y.astype(np.float32)

    rec = _battery_record(
        "lstm",
        {"architecture": "LSTM(1->32) -> head 16->1 on capacity "
                         "sequences",
         "window_cycles": window, "epochs_max": 150, "patience": 15,
         "batch_size": 32, "lr": 0.001, "optimizer": "AdamW",
         "input": "capacity_ah only, standardised with TRAIN "
                  "batteries' statistics; target standardised likewise "
                  "and inverted at prediction"},
        meta, ["capacity_ah (window of 10 discharge cycles)"])
    per_battery, yt_all, yp_all = {}, [], []
    total_train_s, total_pred_s = 0.0, 0.0
    for held in BATTERY_LOBO:
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        train_ids = [b for b in BATTERY_LOBO if b != held]
        tr_df = df[df["battery"].isin(train_ids)]
        te_df = df[df["battery"] == held]
        cap_mean = float(tr_df["capacity_ah"].mean())
        cap_std = float(tr_df["capacity_ah"].std(ddof=0))
        X_parts, y_parts = [], []
        for b in train_ids:
            Xb, yb = windows_for(df[df["battery"] == b])
            X_parts.append(Xb)
            y_parts.append(yb)
        X_tr_raw = np.concatenate(X_parts)
        y_tr_raw = np.concatenate(y_parts)
        y_mean, y_std = float(y_tr_raw.mean()), float(y_tr_raw.std(ddof=0))
        X_tr = ((X_tr_raw - cap_mean) / cap_std).astype(np.float32)
        y_tr = ((y_tr_raw - y_mean) / y_std).astype(np.float32)
        # internal val: last 15% of the pooled train windows (order is
        # battery-major; used for early stopping only)
        n_val = max(8, int(0.15 * len(X_tr)))
        X_va, y_va = X_tr[-n_val:], y_tr[-n_val:]
        X_trf, y_trf = X_tr[:-n_val], y_tr[:-n_val]
        X_te_raw, y_te = windows_for(te_df)
        X_te = ((X_te_raw - cap_mean) / cap_std).astype(np.float32)

        model = BatteryLSTM()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3,
                                weight_decay=1e-4)
        loss_fn = nn.MSELoss()
        dl = DataLoader(TensorDataset(torch.from_numpy(X_trf),
                                      torch.from_numpy(y_trf)),
                        batch_size=32, shuffle=True)
        x_va = torch.from_numpy(X_va)
        y_va_t = torch.from_numpy(y_va)
        best, bad, best_state = float("inf"), 0, None
        t0 = time.perf_counter()
        for epoch in range(150):
            model.train()
            for xb, yb in dl:
                opt.zero_grad()
                loss = loss_fn(model(xb), yb)
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                v = float(torch.sqrt(torch.mean(
                    (model(x_va) - y_va_t) ** 2)))
            if v < best - 1e-4:
                best, bad = v, 0
                best_state = {k: val.clone() for k, val in
                              model.state_dict().items()}
            else:
                bad += 1
                if bad >= 15:
                    break
        total_train_s += time.perf_counter() - t0
        model.load_state_dict(best_state)
        model.eval()
        t0 = time.perf_counter()
        with torch.no_grad():
            pred = model(torch.from_numpy(X_te)).numpy() * y_std + y_mean
        total_pred_s += time.perf_counter() - t0
        pred = np.clip(pred, 0, None)
        per_battery[held] = {"n_cycles_scored": int(len(y_te)),
                             **_reg_metrics(y_te, pred)}
        yt_all.append(y_te)
        yp_all.append(pred)
        print(f"    held out {held}: {per_battery[held]}", flush=True)
    rec["metrics"] = {
        "per_battery": per_battery,
        "pooled": _reg_metrics(np.concatenate(yt_all),
                               np.concatenate(yp_all)),
    }
    n_scored = int(sum(v["n_cycles_scored"] for v in per_battery.values()))
    rec["timing"] = {"train_s": round(total_train_s, 4),
                     "infer_ms_per_row": round(
                         total_pred_s / max(n_scored, 1) * 1000.0, 5)}
    rec["extra"]["window_cycles"] = window
    rec["extra"]["scoring_note"] = ("the first 10 discharge cycles of "
                                    "each held-out battery have no full "
                                    "window and are not scored by the "
                                    "sequence model (the RF scores every "
                                    "cycle)")
    save_record(rec, {"pooled_rmse": round(rec["metrics"]["pooled"]
                                           ["rmse"], 3),
                      "pooled_mae": round(rec["metrics"]["pooled"]
                                          ["mae"], 3)})


# ---------------------------------------------------------- 8) Paderborn RF
def run_paderborn_rf() -> None:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (accuracy_score, classification_report,
                                 confusion_matrix, f1_score)
    from sklearn.model_selection import GroupShuffleSplit

    from platform_lib.adapters.paderborn import LABELS, PaderbornAdapter

    print("=== Paderborn — Task A — RF, measurement-grouped split ===",
          flush=True)
    df, meta = PaderbornAdapter(RAW / "paderborn").load()
    feature_cols = meta["feature_columns"]
    X, y = df[feature_cols], df["label"]
    groups = df["source_file"]
    print(f"  windows: {len(df)}; measurements: {groups.nunique()}; "
          f"per class: {y.value_counts().to_dict()}", flush=True)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    tr_idx, te_idx = next(gss.split(X, y, groups))
    X_train, X_test = X.iloc[tr_idx], X.iloc[te_idx]
    y_train, y_test = y.iloc[tr_idx], y.iloc[te_idx]
    est = RandomForestClassifier(n_estimators=300,
                                 class_weight="balanced",
                                 random_state=SEED, n_jobs=-1)
    rec = base_record("paderborn", "A", "random_forest",
                      {"n_estimators": 300, "class_weight": "balanced",
                       "n_jobs": -1,
                       "features": "per-window time-domain statistics of "
                                   "the vibration_1 channel (64 kHz), "
                                   "same feature space as CWRU"})
    rec["split"] = {
        "protocol": "MEASUREMENT-GROUPED split: GroupShuffleSplit by "
                    "source_file (measurement id), 80/20, seed 42 — no "
                    "measurement file contributes windows to both "
                    "sides. NOTE: there is only ONE physical bearing "
                    "per class in this scope, so train and test still "
                    "share bearings (different measurements of them).",
        "n_train": int(len(X_train)), "n_test": int(len(X_test)),
        "n_measurements_train": int(groups.iloc[tr_idx].nunique()),
        "n_measurements_test": int(groups.iloc[te_idx].nunique()),
        "class_counts_total": {k: int(v) for k, v in
                               y.value_counts().items()},
    }
    rec["preprocessing"] = {
        "features": feature_cols,
        "windowing": "non-overlapping windows of 2048 samples over the "
                     "vibration_1 channel of each 4 s measurement",
        "bearings": meta["bearings"],
        "licence": meta["licence"],
    }
    rec["notes"] = (
        "Caveat (binding): one physical bearing per class (K001 "
        "healthy, KA01 outer-race EDM, KI01 inner-race EDM). The "
        "measurement-grouped split removes the window-level leakage of "
        "a naive split, but the score measures within-bearing "
        "separability across measurements and operating conditions — "
        "NOT generalisation to unseen bearings. CC BY-NC 4.0 data: "
        "non-commercial use only. " + meta.get("data_nature", ""))
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
                        choices=["gru", "cnnlstm", "fd002trees", "ai4i",
                                 "secom", "cwru", "battery", "paderborn",
                                 "all"])
    args = parser.parse_args()
    if args.set in ("gru", "all"):
        run_fd001_gru()
    if args.set in ("cnnlstm", "all"):
        run_fd003_cnn_lstm()
    if args.set in ("fd002trees", "all"):
        run_fd002_trees()
    if args.set in ("ai4i", "all"):
        run_ai4i_svm_mlp()
    if args.set in ("secom", "all"):
        run_secom_anomaly()
    if args.set in ("cwru", "all"):
        run_cwru_recordingwise()
    if args.set in ("battery", "all"):
        run_battery_rf()
        run_battery_lstm()
    if args.set in ("paderborn", "all"):
        run_paderborn_rf()
    print("done.", flush=True)


if __name__ == "__main__":
    main()
