"""PREDICAI — classical baselines on the exact Part 3 protocol (Part 5).

Mean predictor and Ridge regression on flattened windows, evaluated with
the same official protocol as the LSTM: test = last window per engine
(n=100), validation = all windows. Labels are the capped (125) labels,
identical to the shipped model's evaluation.

Run:  PYTHONPATH=ml/src python ml/scripts/run_baselines.py
Out:  ml/artifacts/baselines.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from predicai_ml.data import load_fd001
from predicai_ml.evaluate import evaluate_all
from predicai_ml.preprocessing import run_pipeline

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    train, test, rul = load_fd001(ROOT / "ml" / "data" / "raw")
    res = run_pipeline(train, test, rul)  # cap 125, seed 42 — shipped protocol

    out: dict = {"protocol": "cap=125 labels, seed 42 split, test=last window/engine"}

    # --- Mean predictor: always predict the training-label mean ---
    mean_pred = float(res.y_train.mean())
    for name, y in (("val_all_windows", res.y_val), ("test_last_window", res.y_test_last)):
        pred = np.full_like(y, mean_pred)
        out.setdefault("mean_predictor", {})[name] = evaluate_all(y, pred)
    out["mean_predictor"]["constant_value"] = mean_pred

    # --- Ridge on flattened windows ---
    x_tr = res.X_train.reshape(len(res.X_train), -1)
    x_va = res.X_val.reshape(len(res.X_val), -1)
    x_te = res.X_test_last.reshape(len(res.X_test_last), -1)
    ridge = Ridge(alpha=1.0)
    ridge.fit(x_tr, res.y_train)
    out["ridge_flattened"] = {
        "val_all_windows": evaluate_all(res.y_val, np.clip(ridge.predict(x_va), 0, None)),
        "test_last_window": evaluate_all(res.y_test_last, np.clip(ridge.predict(x_te), 0, None)),
        "alpha": 1.0,
    }

    path = ROOT / "ml" / "artifacts" / "baselines.json"
    path.write_text(json.dumps(out, indent=2))
    for model in ("mean_predictor", "ridge_flattened"):
        t = out[model]["test_last_window"]
        print(f"{model:18s} TEST  RMSE {t['rmse']:7.3f}  MAE {t['mae']:6.3f}  NASA {t['nasa_score']:9.2f}")
    print("saved:", path)


if __name__ == "__main__":
    main()
