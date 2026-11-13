"""Generate the self-contained Kaggle training notebook for PREDICAI.

The notebook mirrors ml/src/predicai_ml (data, preprocessing, model,
evaluate) so the owner can run the full Part 4 training on Kaggle in the
evening and bring back: best_model.pt, cmapss_lstm.onnx,
preprocessing.json, metrics.json, plots.

Run:  python ml/scripts/make_kaggle_notebook.py
Out:  notebooks/PREDICAI_Kaggle_Training.ipynb
"""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "notebooks" / "PREDICAI_Kaggle_Training.ipynb"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.strip("\n").splitlines(keepends=True)}


CELLS = [
    md("""# PREDICAI — Kaggle Training Notebook (Part 4)
**Predictive Intelligence using AI · NASA C-MAPSS FD001 · LSTM RUL Regression**

Run order: **Runtime → Run All**. Pehle Settings me **Accelerator = GPU** on kar lena (fast hoga), warna CPU par bhi chalega, bas slow.

Ye notebook repo ke `ml/src/predicai_ml` code ko mirror karta hai — same preprocessing, same model, same metrics. Kuch bhi alag nahi hai.

**Run ke baad Output se ye files download karni hain (bahut important):**
`best_model.pt`, `cmapss_lstm.onnx`, `preprocessing.json`, `metrics.json`, `training_curves.png`, `pred_vs_actual.png`
"""),
    code("""# ---- Cell 1: setup ----
import math, json, random, zipfile, urllib.request
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEVICE, "| torch:", torch.__version__)

CFG = dict(seq_len=30, rul_cap=125, val_fraction=0.2,
           hidden_1=128, hidden_2=64, head_dim=32, dropout=0.2,
           epochs=60, batch_size=64, lr=1e-3, weight_decay=1e-4,
           grad_clip=1.0, patience=10, opset=17)
WORK = Path("/kaggle/working")
"""),
    code("""# ---- Cell 2: FD001 data dhundho ya download karo ----
# Pehle Kaggle input me dekho (dataset attach kiya ho to), warna NASA se download.
candidates = list(Path("/kaggle/input").rglob("train_FD001.txt")) if Path("/kaggle/input").exists() else []
if candidates:
    RAW = candidates[0].parent
    print("data mila:", RAW)
else:
    RAW = Path("/kaggle/working/raw"); RAW.mkdir(parents=True, exist_ok=True)
    url = "https://data.nasa.gov/docs/legacy/CMAPSSData.zip"
    print("download:", url)
    urllib.request.urlretrieve(url, "/kaggle/working/CMAPSSData.zip")
    with zipfile.ZipFile("/kaggle/working/CMAPSSData.zip") as z:
        for n in ["train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt"]:
            z.extract(n, RAW)
    print("downloaded to", RAW)

COLUMNS = ["unit", "cycle", "setting_1", "setting_2", "setting_3"] + [f"sensor_{i}" for i in range(1, 22)]
def load_file(p):
    df = pd.read_csv(p, sep=r"\\s+", header=None, names=COLUMNS)
    df["unit"] = df["unit"].astype(int); df["cycle"] = df["cycle"].astype(int)
    return df

train_df = load_file(RAW / "train_FD001.txt")
test_df = load_file(RAW / "test_FD001.txt")
true_rul = np.loadtxt(RAW / "RUL_FD001.txt", dtype=int)
print("train", train_df.shape, "| test", test_df.shape, "| RUL", true_rul.shape)
assert train_df.shape == (20631, 26) and test_df.shape == (13096, 26)
"""),
    code("""# ---- Cell 3: preprocessing (repo ke preprocessing.py ka exact mirror) ----
FEATURES = ["sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_8", "sensor_9",
            "sensor_11", "sensor_12", "sensor_13", "sensor_14", "sensor_15",
            "sensor_17", "sensor_20", "sensor_21"]   # 14 sensors — constants + sensor_6 dropped (see DATASET.md)

def add_train_rul(df):
    out = df.copy()
    out["rul"] = (out.groupby("unit")["cycle"].transform("max") - out["cycle"]).astype(int)
    return out

def add_test_rul(df, rul_arr):
    out = df.copy()
    mx = out.groupby("unit")["cycle"].transform("max")
    last = out["unit"].map(lambda u: int(rul_arr[int(u) - 1]))
    out["rul"] = (last + (mx - out["cycle"])).astype(int)
    return out

def engine_split(units, val_fraction=0.2, seed=42):
    rng = np.random.default_rng(seed)
    u = np.array(sorted(int(x) for x in units)); rng.shuffle(u)
    n_val = int(round(len(u) * val_fraction))
    return sorted(u[n_val:].tolist()), sorted(u[:n_val].tolist())

class Scaler:
    def __init__(self, features, mean, scale):
        self.features, self.mean, self.scale = features, mean, scale
    def transform(self, df):
        out = df.copy()
        out[self.features] = (df[self.features].to_numpy() - self.mean) / self.scale
        return out

def fit_scaler(df, features):
    v = df[features].to_numpy(dtype=np.float64)
    return Scaler(features, v.mean(axis=0), v.std(axis=0, ddof=0))

def build_windows(df, features, seq_len=30):
    xs, ys, es = [], [], []
    for unit, grp in df.sort_values(["unit", "cycle"]).groupby("unit"):
        vals = grp[features].to_numpy(dtype=np.float32)
        lab = grp["rul"].to_numpy(dtype=np.float32)
        if len(grp) < seq_len: continue
        for end in range(seq_len, len(grp) + 1):
            xs.append(vals[end - seq_len:end]); ys.append(lab[end - 1]); es.append(int(unit))
    return np.stack(xs), np.array(ys), np.array(es)

def last_windows(df, features, seq_len=30):
    X, y, e = build_windows(df, features, seq_len)
    last = {}
    for i, eng in enumerate(e): last[int(eng)] = i
    idx = [last[k] for k in sorted(last)]
    return X[idx], y[idx], e[idx]

cap = CFG["rul_cap"]
train_df = add_train_rul(train_df); train_df["rul"] = train_df["rul"].clip(upper=cap)
test_df = add_test_rul(test_df, true_rul); test_df["rul"] = test_df["rul"].clip(upper=cap)

tr_eng, val_eng = engine_split(train_df["unit"].unique(), CFG["val_fraction"], SEED)
tr_part = train_df[train_df["unit"].isin(tr_eng)]
val_part = train_df[train_df["unit"].isin(val_eng)]
scaler = fit_scaler(tr_part, FEATURES)          # scaler SIRF train engines par fit hota hai
tr_part = scaler.transform(tr_part); val_part = scaler.transform(val_part)
test_scaled = scaler.transform(test_df)

X_train, y_train, _ = build_windows(tr_part, FEATURES, CFG["seq_len"])
X_val, y_val, _ = build_windows(val_part, FEATURES, CFG["seq_len"])
X_test, y_test, test_engines = last_windows(test_scaled, FEATURES, CFG["seq_len"])
print("X_train", X_train.shape, "| X_val", X_val.shape, "| X_test(last/engine)", X_test.shape)

prep = {"version": 1, "dataset": "NASA C-MAPSS FD001", "features": FEATURES,
        "num_features": len(FEATURES), "seq_len": CFG["seq_len"], "rul_cap": cap,
        "scaler": {"type": "zscore", "ddof": 0,
                   "mean": [float(x) for x in scaler.mean], "scale": [float(x) for x in scaler.scale]},
        "train_engines": tr_eng, "val_engines": val_eng}
(WORK / "preprocessing.json").write_text(json.dumps(prep, indent=2))
print("preprocessing.json saved")
"""),
    code("""# ---- Cell 4: model (repo ke model.py ka mirror) ----
class LSTMRULRegressor(nn.Module):
    def __init__(self, num_features, hidden_1=128, hidden_2=64, head_dim=32, dropout=0.2):
        super().__init__()
        self.lstm1 = nn.LSTM(num_features, hidden_1, batch_first=True)
        self.drop1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(hidden_1, hidden_2, batch_first=True)
        self.drop2 = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(hidden_2, head_dim), nn.ReLU(), nn.Linear(head_dim, 1))
    def forward(self, x):
        out, _ = self.lstm1(x); out = self.drop1(out)
        out, _ = self.lstm2(out)
        return self.head(self.drop2(out[:, -1, :])).squeeze(-1)

model = LSTMRULRegressor(len(FEATURES), CFG["hidden_1"], CFG["hidden_2"], CFG["head_dim"], CFG["dropout"]).to(DEVICE)
print(model)
print("params:", sum(p.numel() for p in model.parameters()))
"""),
    code("""# ---- Cell 5: training (MSE + early stopping on val RMSE) ----
def make_loader(X, y, shuffle):
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(ds, batch_size=CFG["batch_size"], shuffle=shuffle)

train_loader = make_loader(X_train, y_train, True)
val_loader = make_loader(X_val, y_val, False)
opt = torch.optim.Adam(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3)
loss_fn = nn.MSELoss()

def run_epoch(loader, train_mode):
    model.train() if train_mode else model.eval()
    total, n = 0.0, 0
    ctx = torch.enable_grad() if train_mode else torch.no_grad()
    with ctx:
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            if train_mode: opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            if train_mode:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
                opt.step()
            total += loss.item() * len(yb); n += len(yb)
    return math.sqrt(total / n)   # RMSE of the epoch

best_rmse, best_state, bad_epochs = float("inf"), None, 0
hist = {"train": [], "val": []}
for epoch in range(1, CFG["epochs"] + 1):
    tr = run_epoch(train_loader, True)
    va = run_epoch(val_loader, False)
    sched.step(va)
    hist["train"].append(tr); hist["val"].append(va)
    print(f"epoch {epoch:02d} | train RMSE {tr:.2f} | val RMSE {va:.2f}")
    if va < best_rmse - 1e-4:
        best_rmse, bad_epochs = va, 0
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        torch.save(best_state, WORK / "best_model.pt")
    else:
        bad_epochs += 1
        if bad_epochs >= CFG["patience"]:
            print("early stopping at epoch", epoch); break
print("BEST val RMSE:", round(best_rmse, 3))
model.load_state_dict(torch.load(WORK / "best_model.pt")); model.to(DEVICE)

plt.figure(figsize=(8, 4))
plt.plot(hist["train"], label="train RMSE"); plt.plot(hist["val"], label="val RMSE")
plt.xlabel("epoch"); plt.ylabel("RMSE (cycles)"); plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(WORK / "training_curves.png", dpi=150); plt.show()
"""),
    code("""# ---- Cell 6: evaluation — official protocol (last window per test engine) ----
def rmse(a, b): return float(np.sqrt(np.mean((np.asarray(b) - np.asarray(a)) ** 2)))
def mae(a, b): return float(np.mean(np.abs(np.asarray(b) - np.asarray(a))))
def nasa_score(yt, yp):
    d = np.asarray(yp, dtype=np.float64) - np.asarray(yt, dtype=np.float64)
    return float(np.sum(np.exp(-d[d < 0] / 13.0) - 1.0) + np.sum(np.exp(d[d >= 0] / 10.0) - 1.0))

def predict(X):
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), 512):
            xb = torch.from_numpy(X[i:i + 512]).to(DEVICE)
            outs.append(torch.clamp(model(xb), min=0.0).cpu().numpy())
    return np.concatenate(outs)

pred_val = predict(X_val)
pred_test = predict(X_test)
metrics = {
    "config": CFG, "seed": SEED,
    "val_all_windows": {"rmse": rmse(y_val, pred_val), "mae": mae(y_val, pred_val),
                        "nasa_score": nasa_score(y_val, pred_val), "n": int(len(y_val))},
    "test_last_window_per_engine": {"rmse": rmse(y_test, pred_test), "mae": mae(y_test, pred_test),
                                    "nasa_score": nasa_score(y_test, pred_test), "n": int(len(y_test))},
    "best_val_rmse_during_training": best_rmse,
}
(WORK / "metrics.json").write_text(json.dumps(metrics, indent=2))
print(json.dumps(metrics, indent=2))
np.save(WORK / "test_predictions.npy", np.stack([test_engines, y_test, pred_test], axis=1))

plt.figure(figsize=(6, 6))
plt.scatter(y_test, pred_test, s=14, alpha=0.7)
lim = [0, max(float(y_test.max()), float(pred_test.max())) + 5]
plt.plot(lim, lim, "r--", label="perfect")
plt.xlabel("Actual RUL (cycles)"); plt.ylabel("Predicted RUL (cycles)")
plt.title("PREDICAI FD001 — Test (last window per engine)"); plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(WORK / "pred_vs_actual.png", dpi=150); plt.show()
"""),
    code("""# ---- Cell 7: ONNX export + PyTorch vs ONNX parity check ----
try:
    import onnx, onnxruntime as ort
except ImportError:
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "onnx", "onnxruntime", "onnxscript"], check=True)
    import onnx, onnxruntime as ort

model.eval()
dummy = torch.randn(1, CFG["seq_len"], len(FEATURES), device=DEVICE)
onnx_path = WORK / "cmapss_lstm.onnx"
torch.onnx.export(model, dummy, str(onnx_path),
                  input_names=["input"], output_names=["rul"],
                  dynamic_axes={"input": {0: "batch"}, "rul": {0: "batch"}},
                  opset_version=CFG["opset"], do_constant_folding=True,
                  dynamo=False)  # legacy exporter — torch>=2.9 ka dynamo default onnxscript maangta hai, wo Kaggle par missing tha (13 Nov run me yahi error aaya tha)
onnx.checker.check_model(onnx.load(str(onnx_path)))
sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
sample = X_val[:100]
with torch.no_grad():
    pt = torch.clamp(model(torch.from_numpy(sample).to(DEVICE)), min=0.0).cpu().numpy()
ox = sess.run(None, {"input": sample})[0].reshape(-1)
# NOTE: ONNX graph predicts RAW output (clamp is inference-side); compare raw vs raw:
with torch.no_grad():
    pt_raw = model(torch.from_numpy(sample).to(DEVICE)).cpu().numpy()
max_diff = float(np.max(np.abs(pt_raw - ox)))
print("PyTorch vs ONNX max abs diff:", max_diff)
assert max_diff <= 1e-4, "PARITY FAILED — export dobara check karo"
print("PARITY OK — cmapss_lstm.onnx ready")
"""),
    md("""## Run complete — ab kya download karna hai

Kaggle ke **Output** section se ye 6 files download karo aur Muse ko bhej do:
1. `best_model.pt`
2. `cmapss_lstm.onnx`
3. `preprocessing.json`
4. `metrics.json`
5. `training_curves.png`
6. `pred_vs_actual.png` (+ ho sake to `test_predictions.npy`)

Inhi files se Part 5 (evaluation verify), Part 7 (ONNX/Rust) aage badhega. Metrics jo `metrics.json` me aaye hain wahi asli result hai — wahi README me jayega.
"""),
]


def main() -> None:
    nb = {
        "cells": CELLS,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python", "version": "3.11"}},
        "nbformat": 4, "nbformat_minor": 5,
    }
    # syntax-check every code cell before writing
    for i, cell in enumerate(CELLS):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"cell_{i}", "exec")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(nb, indent=1))
    print(f"Notebook written: {OUT} ({len(CELLS)} cells, all code cells compile OK)")


if __name__ == "__main__":
    main()
