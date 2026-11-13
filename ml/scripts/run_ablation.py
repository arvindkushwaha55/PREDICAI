"""PREDICAI — RUL-cap ablation (Part 5): cap 150 and no-cap vs shipped 125.

The shipped model (cap 125) was trained by the owner on Kaggle (GPU).
These ablation variants are trained LOCALLY on CPU with the identical
config, seed, split and features — only the label cap changes — purely
to answer the plan's open question: does the cap choice matter?
The shipped model is NOT replaced by anything produced here.

Run:  PYTHONPATH=ml/src python ml/scripts/run_ablation.py
Out:  ml/artifacts/ablation_results.json (+ per-cap metrics printed)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from predicai_ml.data import load_fd001
from predicai_ml.evaluate import evaluate_all
from predicai_ml.model import LSTMRULRegressor, predict_rul
from predicai_ml.preprocessing import FEATURES, run_pipeline

ROOT = Path(__file__).resolve().parents[2]
CFG = dict(hidden_1=128, hidden_2=64, head_dim=32, dropout=0.2,
           epochs=60, batch_size=64, lr=1e-3, weight_decay=1e-4,
           grad_clip=1.0, patience=10, seed=42)


def train_one(cap) -> dict:
    torch.manual_seed(CFG["seed"])
    np.random.seed(CFG["seed"])
    train, test, rul = load_fd001(ROOT / "ml" / "data" / "raw")
    res = run_pipeline(train, test, rul, cap=cap)

    ds = TensorDataset(torch.from_numpy(res.X_train), torch.from_numpy(res.y_train))
    dl = DataLoader(ds, batch_size=CFG["batch_size"], shuffle=True)
    x_va = torch.from_numpy(res.X_val)
    x_te = torch.from_numpy(res.X_test_last)

    model = LSTMRULRegressor(len(FEATURES), CFG["hidden_1"], CFG["hidden_2"],
                             CFG["head_dim"], CFG["dropout"])
    opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CFG["epochs"])
    loss_fn = nn.MSELoss()

    best_rmse, best_state, bad_epochs = float("inf"), None, 0
    t0 = time.time()
    for epoch in range(1, CFG["epochs"] + 1):
        model.train()
        for xb, yb in dl:
            opt.zero_grad()
            loss = loss_fn(model(xb).squeeze(-1), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
            opt.step()
        sched.step()
        with torch.no_grad():
            val_rmse = float(torch.sqrt(torch.mean(
                (predict_rul(model, x_va) - torch.from_numpy(res.y_val)) ** 2)))
        print(f"  cap={cap} epoch {epoch:02d}  val RMSE {val_rmse:.3f}", flush=True)
        if val_rmse < best_rmse - 1e-4:
            best_rmse, bad_epochs = val_rmse, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= CFG["patience"]:
                print(f"  cap={cap} early stop at epoch {epoch}", flush=True)
                break
    model.load_state_dict(best_state)
    with torch.no_grad():
        test_pred = predict_rul(model, x_te).numpy()
        val_pred = predict_rul(model, x_va).numpy()
    return {
        "cap": cap,
        "device": "cpu (local ablation run)",
        "minutes": round((time.time() - t0) / 60, 1),
        "val_all_windows": evaluate_all(res.y_val, val_pred),
        "test_last_window": evaluate_all(res.y_test_last, test_pred),
    }


def main() -> None:
    results = {"question": "does the RUL label cap change test performance?",
               "reference_cap_125": json.loads(
                   (ROOT / "ml" / "artifacts" / "metrics.json").read_text())[
                   "test_last_window_per_engine"] | {"source": "owner Kaggle GPU run (shipped model)"},
               "variants": []}
    for cap in (150, None):
        print(f"=== training cap={cap} ===", flush=True)
        results["variants"].append(train_one(cap))
        path = ROOT / "ml" / "artifacts" / "ablation_results.json"
        path.write_text(json.dumps(results, indent=2))
        v = results["variants"][-1]["test_last_window"]
        print(f"cap={cap} TEST RMSE {v['rmse']:.3f} MAE {v['mae']:.3f} NASA {v['nasa_score']:.2f}", flush=True)
    print("saved: ml/artifacts/ablation_results.json", flush=True)


if __name__ == "__main__":
    main()
