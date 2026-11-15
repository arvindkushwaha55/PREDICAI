"""PREDICAI platform — benchmarking wave 10 (2026-11-15).

Two items, both closing entries from the wave-8/9 "still open" lists:

1. PER-REGIME NORMALISATION for the 6-condition C-MAPSS subsets.
   Wave 2 scored FD002/FD004 with one global z-score scaler; this wave
   re-runs Ridge (FD002 + FD004) and the LSTM (FD002 only — CPU
   budget; the FD004 LSTM is deliberately not re-run and the record
   set says so by its absence) with regime_pipeline.py, which changes
   ONLY the scaling: KMeans k=6 regimes derived from the TRAIN
   split's settings, one sensor scaler per regime, settings keep the
   global scaler. Every record carries the wave-2 global-norm numbers
   as its comparison block, deltas included — the wave-10 number
   without its wave-2 baseline is not reportable.

2. PADERBORN REAL-DAMAGE BEARINGS. Waves 3-4 used the artificial
   EDM-seeded bearings KA01/KI01. The dataset also contains REAL
   fatigue damage from accelerated lifetime tests; the picks here
   were verified from the OFFICIAL per-bearing damage-profile PDFs
   shipped inside each official archive (Lessmeier, KAt Paderborn),
   read verbatim — not inferred from bearing codes:
     KA04 — 1 damage: mode "fatigue", symptom "Pitting", component
            OR, method "lifetime test", extent 1, 2 x 3 mm.
     KI04 — primary damage: mode "fatigue" (below the surface),
            symptom "Pitting", component IR, method "lifetime test",
            extent 1, 2 x 1 mm; PLUS a documented secondary damage
            (plastic deformation by foreign objects, OR indentations)
            — stated in the record, not hidden. (KI14's profile was
            read too and shows the same primary+secondary pattern;
            KI04 was downloaded first and kept.)
   RF on the same 10 time-domain features, same measurement-grouped
   split as wave 4, classes {Normal (K001), InnerRace (KI04),
   OuterRace (KA04)}.

Usage (from the platform/ directory):
    ../.venv/bin/python -m platform_lib.benchmarking.run_wave10 --set fd002-ridge
    ../.venv/bin/python -m platform_lib.benchmarking.run_wave10 --set fd004-ridge
    ../.venv/bin/python -m platform_lib.benchmarking.run_wave10 --set fd002-lstm
    ../.venv/bin/python -m platform_lib.benchmarking.run_wave10 --set paderborn-real

Seed is 42 everywhere.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

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


# ------------------------------------------------------- C-MAPSS subsets
def _subset_data_perregime(subset: str):
    from platform_lib.adapters.cmapss import CMapssAdapter
    from platform_lib.regime_pipeline import build_rul_data_perregime

    adapter = CMapssAdapter(RAW / "cmapss" / "extracted", subset=subset)
    train, meta = adapter.load()
    test, rul = adapter.load_test()
    data = build_rul_data_perregime(train, test, rul, cap=125, seq_len=30,
                                    seed=SEED)
    return meta, data


def _wave2_test_metrics(subset: str, wave2_model: str) -> dict:
    path = (PLATFORM_DIR / "experiments" /
            f"cmapss_{subset.lower()}__taskB__{wave2_model}__s42.json")
    rec = json.loads(path.read_text())
    m = rec["metrics"]["test_last_window_per_engine"]
    return {"record": path.name,
            "rmse": m["rmse"], "mae": m["mae"], "nasa_score": m["nasa_score"]}


def _perregime_record(subset: str, model: str, params: dict, meta: dict,
                      data, wave2_model: str) -> dict:
    key = f"cmapss_{subset.lower()}"
    rec = base_record(key, "B", model, params)
    rec["split"] = {
        "protocol": "engine-wise 80/20 train/val (seed 42) + official "
                    "provided test set, last window per engine — "
                    "IDENTICAL to wave 2; the ONLY change vs wave 2 is "
                    "the normalisation (per-regime sensor scaling)",
        "n_train_engines": len(data.train_engines),
        "n_val_engines": len(data.val_engines),
        "n_train_windows": int(len(data.X_train)),
        "n_val_windows": int(len(data.X_val)),
        "n_test_engines": int(len(data.X_test_last)),
    }
    ri = data.regime_info
    rec["preprocessing"] = {
        "features": data.features,
        "n_features": len(data.features),
        "feature_derivation": "candidates = 3 settings + 21 sensors; "
                              "dropped train-split std < 0.005 (same "
                              "rule as wave 2, see rul_pipeline.py)",
        "dropped_candidates": data.dropped,
        "rul_cap": 125, "seq_len": 30,
        "scaling": "PER-REGIME z-score (ddof=0) for the 20 sensor "
                   "features: KMeans k=6 regimes derived from the "
                   "TRAIN split's settings only; one scaler per regime "
                   "fit on that regime's train rows; val/test rows "
                   "assigned to the nearest train centroid. The 3 "
                   "setting features keep the wave-2 global train-split "
                   "z-score (within a regime a setting is near-constant "
                   "by construction, so a per-regime z-score for it is "
                   "undefined).",
        "regimes": {
            "k": ri["k"],
            "method": ri["method"],
            "train_sizes_rows": ri["train_sizes"],
            "val_sizes_rows": ri["val_sizes"],
            "test_sizes_rows": ri["test_sizes"],
            "centroids_raw_settings": ri["centroids_raw_settings"],
            "zero_std_fallbacks": ri["zero_std_fallbacks"],
        },
    }
    rec["notes"] = (
        f"{subset} = {meta['operating_conditions']} operating "
        f"conditions, {meta['fault_modes']} fault mode(s). Wave-10 "
        f"per-regime normalisation re-run of the wave-2 "
        f"{wave2_model} experiment; split, features, cap, windowing "
        f"and model are unchanged — only the scaling differs, so the "
        f"delta vs the comparison block is the normalisation effect "
        f"as measured. " + meta.get("data_nature", ""))
    rec["extra"] = {"candidate_train_stds": data.candidate_stds}
    rec["_wave2_model"] = wave2_model  # popped before save; see _finish
    return rec


def _finish(rec: dict, subset: str) -> None:
    wave2_model = rec.pop("_wave2_model")
    base = _wave2_test_metrics(subset, wave2_model)
    new = rec["metrics"]["test_last_window_per_engine"]
    rec["extra"]["comparison_wave2_global_normalisation"] = {
        "wave2_global": base,
        "wave10_perregime": {"rmse": new["rmse"], "mae": new["mae"],
                             "nasa_score": new["nasa_score"]},
        "delta_rmse": new["rmse"] - base["rmse"],
        "delta_mae": new["mae"] - base["mae"],
        "delta_nasa": new["nasa_score"] - base["nasa_score"],
        "delta_rmse_pct": 100.0 * (new["rmse"] / base["rmse"] - 1.0),
    }
    save_record(rec, {"test_rmse": round(new["rmse"], 3),
                      "test_mae": round(new["mae"], 3),
                      "test_nasa": round(new["nasa_score"], 1)})


def run_subset_ridge_perregime(subset: str) -> None:
    from sklearn.linear_model import Ridge

    from predicai_ml.evaluate import evaluate_all

    print(f"=== C-MAPSS {subset} — Task B — Ridge, PER-REGIME scaling ===",
          flush=True)
    meta, data = _subset_data_perregime(subset)
    ri = data.regime_info
    print(f"  features ({len(data.features)}); regime train sizes: "
          f"{ri['train_sizes']}", flush=True)
    print(f"  regime centroids (settings): "
          f"{[tuple(c.values()) for c in ri['centroids_raw_settings']]}",
          flush=True)
    Xtr = data.X_train.reshape(len(data.X_train), -1)
    Xva = data.X_val.reshape(len(data.X_val), -1)
    Xte = data.X_test_last.reshape(len(data.X_test_last), -1)

    rec = _perregime_record(
        subset, "ridge_perregime",
        {"alpha": 1.0,
         "input": "flattened (30 x F) windows, predictions clipped "
                  "at >= 0 — identical to wave 2"},
        meta, data, wave2_model="ridge_flat")
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
    _finish(rec, subset)


def run_subset_lstm_perregime(subset: str) -> None:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    from predicai_ml.evaluate import evaluate_all
    from predicai_ml.model import LSTMRULRegressor, predict_rul

    print(f"=== C-MAPSS {subset} — Task B — LSTM, PER-REGIME scaling ===",
          flush=True)
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
        print(f"  {subset.lower()}-lstm-perregime epoch {epoch:02d}  "
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
    _finish(rec, subset)


# ------------------------------------------------- Paderborn real damage
REAL_BEARINGS = ("K001", "KA04", "KI04")

BEARING_DOCUMENTATION = {
    "K001": {
        "bearing": "K001", "label": "Normal",
        "damage_origin": "none (healthy bearing)",
        "documented_damage": None,
        "archive_sha256":
            "0f119ebdb28fb2f4d9fac1beb1319429f63f7ae1256c23c872f280f3560918e5",
    },
    "KA04": {
        "bearing": "KA04", "label": "OuterRace",
        "damage_origin": "real (fatigue pitting, accelerated lifetime test)",
        "documented_damage": {
            "number_of_damages": 1,
            "mode": "fatigue",
            "symptom": "Pitting",
            "component": "OR (outer ring)",
            "position_of_damage": "raceway",
            "damage_combination": "S (single damage)",
            "length_mm": 2, "width_mm": 3,
            "extent_of_damage": 1,
            "characteristic_of_damage": "single point",
            "damage_method": "lifetime test",
            "cause_of_damage": "operating conditions and lubricant "
                               "(overload, wrong viscosity, contamination)",
        },
        "archive_sha256":
            "6fc317157e2cd4cb55bc54d6fac1b1753f8ab31abcccd1ac68cbfb78e5e8299d",
    },
    "KI04": {
        "bearing": "KI04", "label": "InnerRace",
        "damage_origin": "real (fatigue pitting, accelerated lifetime "
                         "test; primary damage)",
        "documented_damage": {
            "number_of_damages": 2,
            "damage_1_primary": {
                "mode": "fatigue (below the surface)",
                "symptom": "Pitting",
                "component": "IR (inner ring)",
                "position_of_damage": "raceway",
                "damage_combination": "M (multiple damage)",
                "length_mm": 2, "width_mm": 1,
                "extent_of_damage": 1,
                "characteristic_of_damage": "single point",
                "damage_method": "lifetime test",
            },
            "damage_2_secondary": {
                "mode": "plastic deformation (by foreign objects)",
                "symptom": "particle-caused indentations",
                "component": "OR (outer ring)",
                "position_of_damage": "raceway",
                "extent_of_damage": "n/a",
                "characteristic_of_damage": "single point",
                "damage_method": "lifetime test",
            },
            "classification_note": "the bearing is categorised by its "
                                   "primary damage (inner-race fatigue "
                                   "pitting); the secondary outer-race "
                                   "indentations are part of its "
                                   "official profile and are stated "
                                   "here, not hidden",
        },
        "archive_sha256":
            "8fde861a9ceac4b41fe8a030b4c5bbb4b9b9af0590d9338a8de0150b4661be77",
    },
}

DOC_SOURCE = ("official per-bearing damage-profile PDF shipped inside "
              "each official .rar archive (e.g. KA04/KA04.pdf), "
              "'Profile of rolling bearing damage', Dipl.-Ing. "
              "Christian Lessmeier, Chair of Design and Drive "
              "Technology, Paderborn University — fields copied "
              "verbatim, read 2026-11-15")


def run_paderborn_real() -> None:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (accuracy_score, classification_report,
                                 confusion_matrix, f1_score)
    from sklearn.model_selection import GroupShuffleSplit

    from platform_lib.adapters.paderborn import LABELS, PaderbornAdapter

    print("=== Paderborn REAL damage — Task A — RF, grouped split ===",
          flush=True)
    df, meta = PaderbornAdapter(RAW / "paderborn",
                                bearings=REAL_BEARINGS).load()
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
    rec = base_record("paderborn_real", "A", "random_forest",
                      {"n_estimators": 300, "class_weight": "balanced",
                       "n_jobs": -1,
                       "features": "per-window time-domain statistics "
                                   "of the vibration_1 channel (64 kHz) "
                                   "— same 10 features as the wave-4 "
                                   "artificial-damage run"})
    rec["split"] = {
        "protocol": "MEASUREMENT-GROUPED split: GroupShuffleSplit by "
                    "source_file (measurement id), 80/20, seed 42 — "
                    "identical protocol to wave 4; the ONLY change vs "
                    "wave 4 is the bearing set (real fatigue damage "
                    "instead of artificial EDM damage). NOTE: one "
                    "physical bearing per class, so train and test "
                    "share bearings (different measurements of them).",
        "n_train": int(len(X_train)), "n_test": int(len(X_test)),
        "n_measurements_train": int(groups.iloc[tr_idx].nunique()),
        "n_measurements_test": int(groups.iloc[te_idx].nunique()),
        "class_counts_total": {k: int(v) for k, v in
                               y.value_counts().items()},
    }
    rec["preprocessing"] = {
        "features": feature_cols,
        "windowing": "non-overlapping windows of 2048 samples over "
                     "the vibration_1 channel of each 4 s measurement",
        "bearings": meta["bearings"],
        "licence": meta["licence"],
    }
    rec["notes"] = (
        "Caveat (binding): one physical bearing per class (K001 "
        "healthy, KA04 outer-race REAL fatigue pitting, KI04 "
        "inner-race REAL fatigue pitting as its primary damage). The "
        "measurement-grouped split removes window-level leakage, but "
        "the score measures within-bearing separability across "
        "measurements and operating conditions — NOT generalisation "
        "to unseen bearings, and NOT an artificial-vs-real damage "
        "comparison (different bearings, not the same bearing before/"
        "after). KI04's official profile documents a SECONDARY "
        "outer-race damage (particle-caused indentations) alongside "
        "its primary inner-race pitting; it is recorded in "
        "extra.bearing_documentation, not hidden. CC BY-NC 4.0 data: "
        "non-commercial use only. " + meta.get("data_nature", ""))
    rec["extra"] = {
        "documentation_source": DOC_SOURCE,
        "bearing_documentation": [BEARING_DOCUMENTATION[c]
                                  for c in REAL_BEARINGS],
        "comparison_wave4_artificial": {
            "record": "paderborn__taskA__random_forest__s42",
            "note": "wave-4 run on the ARTIFICIAL EDM bearings "
                    "(KA01/KI01) under the identical protocol; shown "
                    "for context only — different physical bearings, "
                    "so this is not a controlled artificial-vs-real "
                    "comparison",
        },
    }
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
    # Fill the wave-4 context numbers now that the record exists.
    w4 = json.loads((PLATFORM_DIR / "experiments" /
                     "paderborn__taskA__random_forest__s42.json").read_text())
    rec["extra"]["comparison_wave4_artificial"].update({
        "accuracy": w4["metrics"]["accuracy"],
        "f1_macro": w4["metrics"]["f1_macro"],
        "wave10_real_accuracy": rec["metrics"]["accuracy"],
        "wave10_real_f1_macro": rec["metrics"]["f1_macro"],
    })
    save_record(rec, {"accuracy": round(rec["metrics"]["accuracy"], 4),
                      "f1_macro": round(rec["metrics"]["f1_macro"], 4)})


# ------------------------------------------------------------------- CLI
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", required=True,
                        choices=["fd002-ridge", "fd004-ridge",
                                 "fd002-lstm", "paderborn-real",
                                 "regimes", "all"])
    args = parser.parse_args()
    if args.set in ("fd002-ridge", "regimes", "all"):
        run_subset_ridge_perregime("FD002")
    if args.set in ("fd004-ridge", "regimes", "all"):
        run_subset_ridge_perregime("FD004")
    if args.set in ("fd002-lstm", "regimes", "all"):
        run_subset_lstm_perregime("FD002")
    if args.set in ("paderborn-real", "all"):
        run_paderborn_real()
    print("done.", flush=True)


if __name__ == "__main__":
    main()
