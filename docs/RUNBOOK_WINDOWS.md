# PREDICAI — Windows 11 Runbook (PowerShell)

Everything below runs on the owner's laptop. Reference environment:
Windows 11 + PowerShell. Paths assume the repo is cloned at
`C:\Users\<you>\PREDICAI` — adjust once, consistently.

## 0. Prerequisites

- Python 3.12 (python.org installer, "Add python.exe to PATH" ticked)
- Rust stable via rustup (https://rustup.rs) — for the edge API
- Node.js 20+ — for the dashboard
- Git for Windows

## 1. Clone (private repo — sign in as arvindkushwaha55)

```powershell
git clone https://github.com/arvindkushwaha55/PREDICAI.git
cd PREDICAI
```

## 2. Python environment + data

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r ml\requirements.txt   # if present; otherwise: pip install pandas numpy scikit-learn pytest onnx onnxruntime torch --index-url https://download.pytorch.org/whl/cpu
```

Download FD001 (NASA, free, no account): the notebook in `notebooks\` does
this itself on Kaggle; locally, download `CMAPSSData.zip` from
data.nasa.gov ("C-MAPSS" dataset page) and extract the `train_FD001.txt`,
`test_FD001.txt`, `RUL_FD001.txt` files into `ml\data\raw\`.
**Do not commit the raw data** — its licence does not allow redistribution
(the repo's `.gitignore` already excludes it).

## 3. The model

Already trained (Kaggle, 13 Nov 2026) — the artifacts are IN the repo:
`ml\artifacts\cmapss_lstm.onnx`, `best_model.pt`, `metrics.json`,
`preprocessing.json`. Nothing to retrain for the demo.
Re-training from scratch: follow `docs\KAGGLE_GUIDE.md`.

## 4. Rust edge API

The `ort` crate is pinned (`=2.0.0-rc.15`) with the `load-dynamic`
feature, so the ONNX Runtime library is loaded from the Python package:

```powershell
cd edge
$env:ORT_DYLIB_PATH = "..\.venv\Lib\site-packages\onnxruntime\capi\onnxruntime.dll"
cargo run            # serves http://localhost:8080 (PREDICAI_PORT overrides)
```

Check: `Invoke-RestMethod http://localhost:8080/health` →
`model_loaded` must be `true`.

Tests: `cargo test` (the parity test needs `ORT_DYLIB_PATH` set, see above).

## 5. Simulator (FD001 replay → API)

```powershell
cd ..\simulator
..\.venv\Scripts\python simulator.py --engine 49 --api http://localhost:8080
```

This replays RECORDED test data (labelled simulated replay — not live
sensors) and the API logs every prediction to its SQLite store.

## 6. Dashboard

```powershell
cd ..\dashboard
npm install
npm run dev          # http://localhost:5173, /api proxied to :8080
```

Start the Rust API (step 4) first if you want the Live API panel to work;
without it the dashboard still shows all replay data and honestly marks
the API OFFLINE. (A static `npm run build` output cannot reach the API —
see dashboard/README.md for why.)

## 7. Full test sweep

```powershell
..\.venv\Scripts\python -m pytest ml\tests maintenance simulator   # from repo root, venv active
cd edge; cargo test
cd ..\dashboard; npm run build
```
