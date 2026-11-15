<div align="center">

# PREDICAI

### Predictive Intelligence using Artificial Intelligence

**Predict Today. Prevent Tomorrow.**

*An AI-powered predictive maintenance system that learns machine degradation
from sensor data, estimates Remaining Useful Life (RUL), and serves
predictions from a high-performance Rust edge engine.*

![Status](https://img.shields.io/badge/status-in%20development-yellow)
![Dataset](https://img.shields.io/badge/dataset-NASA%20C--MAPSS%20FD001-blue)
![Model](https://img.shields.io/badge/model-LSTM%20%7C%20PyTorch-orange)
![Serving](https://img.shields.io/badge/serving-Rust%20%7C%20ONNX%20Runtime-red)
![License](https://img.shields.io/badge/license-MIT-green)

**Live dashboard: [arvindkushwaha55.github.io/PREDICAI](https://arvindkushwaha55.github.io/PREDICAI/)**
(the hosted build runs entirely on bundled data; the Live API panel needs the Rust edge API running locally)

</div>

---

> **Honesty note (read first):** PREDICAI is built on the NASA C-MAPSS
> *simulated* turbofan benchmark — there is no physical turbine. Sensor
> "live" data in the demo is a replay of real FD001 test data; maintenance
> records shown in the demo are **synthetic and labelled as such**. Model
> metrics below are filled in only from actual training runs — until then
> they read `—`. RUL is always measured in **operating cycles**.

## The problem

Machines that run continuously — turbines, motors, engines — fail, and
unplanned failure means downtime, expensive repairs and lost production.
Fixed maintenance schedules fail both ways: too late (the machine breaks
first) or too early (healthy machines get serviced anyway). PREDICAI
estimates *how much useful life is left*, so maintenance can be planned
from evidence instead of a calendar.

## What PREDICAI does

1. **Learns degradation** — a PyTorch LSTM reads 30-cycle windows of
   multivariate sensor data from NASA C-MAPSS FD001 and regresses RUL.
2. **Explains itself** — sensor-importance analysis (permutation importance
   + Integrated Gradients) shows which sensors drive each estimate.
3. **Deploys to the edge** — the model is exported to ONNX and served by a
   Rust API (Axum + ONNX Runtime): validate → normalize → infer → respond.
4. **Tracks maintenance** — a SQLite maintenance module records service
   history per machine and combines it with predicted RUL to produce
   *reasoned* recommendations (policy-based, clearly labelled).
5. **Demonstrates live** — a simulator replays real FD001 test engines
   cycle-by-cycle into the Rust API; a React dashboard visualizes sensors,
   RUL, history and recommendations.

## System architecture

```
NASA C-MAPSS FD001
      │
      ▼
┌─────────────────── PYTHON ML LAB ───────────────────┐
│ EDA → Preprocessing → RUL labels → Sliding windows  │
│ → LSTM training → Evaluation → Explainability       │
└──────────────────────┬──────────────────────────────┘
                       │  cmapss_lstm.onnx
                       │  + preprocessing.json (shared contract)
                       ▼
┌─────────────────── RUST EDGE ENGINE ────────────────┐
│ Axum API · ONNX Runtime (ort) · input validation    │
│ GET /health        POST /predict_rul                │
└──────────┬───────────────────────────▲──────────────┘
           │                           │
           ▼                           │ replay
┌──────────────────┐        ┌──────────┴───────────────┐
│ REACT DASHBOARD  │        │ SENSOR SIMULATOR         │
│ sensors · RUL ·  │        │ FD001 test-engine replay │
│ history · XAI    │        │ rolling 30-cycle window  │
└────────▲─────────┘        └──────────────────────────┘
         │
┌────────┴────────────────────────────────────────────┐
│ MAINTENANCE MODULE (SQLite)                         │
│ machines · service records · policy recommendations │
└─────────────────────────────────────────────────────┘
```

## Model performance (FD001, official test protocol)

> Filled from Phase A4 evaluation runs only. Protocol: last window per test
> engine, RUL cap and config recorded alongside the numbers.

| Model | RMSE (cycles) | MAE (cycles) | NASA PHM08 Score |
|---|---|---|---|
| Mean predictor (baseline) | 40.69 | 35.00 | 19,887.2 |
| Ridge, flattened windows (baseline) | 15.31 | 12.43 | 407.8 |
| **PREDICAI LSTM** | **13.73** | **9.89** | **431.3** |

*Baselines measured 13 Nov 2026 on the identical protocol (cap-125 labels,
seed-42 split). Read honestly: the LSTM wins clearly on RMSE and MAE, but
Ridge's NASA score is slightly better (407.8 vs 431.3) — the asymmetric
NASA penalty punishes the LSTM's occasional late (over-life) predictions
on this test set more than Ridge's. Both facts stay in the table.*

*Measured 13 Nov 2026 — Kaggle training run (seed 42, RUL cap 125, 14 sensors),
independently re-verified from the raw predictions file: recomputed metrics
match `metrics.json` exactly. Validation (all windows, n=3,272): RMSE 13.01,
MAE 9.02. Error is smallest where it matters most — true RUL 0–25 band MAE
is just 2.92 cycles. Baselines: see the table (measured, same protocol).
RUL-cap ablation — MEASURED 13 Nov 2026 (variants retrained locally on
CPU, identical config/seed/split; only the label cap changed):*

| Label cap | Test RMSE | Test MAE | NASA |
|---|---|---|---|
| **125 (shipped)** | **13.73** | **9.89** | **431.3** |
| 150 | 16.51 | 12.30 | 941.3 |
| none | 26.24 | 19.30 | 23,109.4 |

*The cap is not a detail: removing it nearly doubles the error. 125 stays.
Full data: `ml/artifacts/ablation_results.json`.*

*Cross-stack parity (all measured): PyTorch ↔ ONNX max diff 1.53e-05;
Rust API ↔ Kaggle PyTorch prediction for test engine 1 identical to 5
decimal places (117.60932 cycles).*

*Inference benchmark — MEASURED 13 Nov 2026 (Rust API release build, this
machine, localhost, 1,000 sequential requests, each including prediction +
SQLite log write): **p50 1.36 ms · p95 2.23 ms** (mean 1.51 ms). Local
development-machine numbers, not a production guarantee.*

## Platform (Phase 8 expansion)

Alongside the FD001 system above — which it does not modify — `platform/`
generalises PREDICAI into a multi-dataset, multi-model research platform
over three task families: **A** failure/fault classification, **B** RUL
regression, **C** anomaly detection. A central registry tracks 20 dataset
entries; **10 are integrated** (adapters + local data + experiment
records): C-MAPSS FD001–FD004, AI4I 2020, SECOM, CWRU, Paderborn
(artificial EDM + real fatigue-damage bearings), IMS (Tests 1–3) and
the NASA Li-ion Battery set. **50
experiment records** (Task A 23 · B 21 · C 6, all seed 42) live in
`platform/experiments/`, each with its split protocol stated — splits are
asset/recording-wise wherever windows from one machine could otherwise
leak across train/test.

Headline results, stated as measured (full detail in
[`platform/EXPERIMENTS_REPORT.md`](platform/EXPERIMENTS_REPORT.md)):

- The shipped FD001 LSTM (RMSE **13.73**) remains the deployed model. A
  GRU on the identical protocol scored **13.22** — one seed, marginally
  ahead, recorded as such and not swapped in on that evidence.
- FD003 CNN-LSTM: RMSE **13.17**, ahead of the FD003 LSTM (15.04).
- CWRU bearing diagnosis: the window-split Random Forest scores 1.000 —
  but that measures *within-recording* separability. The honest
  recording-wise re-evaluation scores **0.665** on time-domain
  features — and **0.802** with Wave 12's order-domain features (see
  below); both numbers are kept, the caveat travels with them.
- Cross-dataset transfer mostly fails, and is documented failing:
  FD001→FD003 RMSE 48.80 (unseen fault mode) vs FD003→FD001 15.82;
  FD002→FD004 51.38 vs FD004→FD002 18.07; CWRU↔Paderborn accuracy
  0.253 / 0.125 — fault *location* does not transfer across rigs on
  time-domain features.
- IMS Test 1 anomaly detection is reported as **alarm timing only**
  (no per-snapshot labels exist): first sustained alarms 337 and 723
  snapshots before the end of the run on the two bearings with
  documented teardown defects, none on the two without.
- Waves 9–10, reported as measured: adding spectral + envelope
  features did **not** fix bearing transfer (CWRU recording-wise
  0.679 vs 0.665; both cross-rig transfers still fail). IMS Tests 2–3
  ran under the identical protocol — Test 2's failed bearing alarmed
  first (452 snapshots early; healthy bearings also alarmed later),
  and Test 3's failed bearing was **missed entirely**; the protocol
  was not re-tuned after seeing that. Per-regime normalisation gives
  FD002 a new best: LSTM RMSE **14.05** (global normalisation: 16.24).
  Paderborn *real* fatigue-damage bearings (KA04/KI04) score **0.9977**
  accuracy under the identical grouped protocol — within-bearing
  separability with one physical bearing per class, not
  unseen-bearing generalisation.
- Wave 12, reported as measured: the FD004 per-regime LSTM scores
  RMSE **15.37** — FD004's new best, and it retakes the NASA-score
  lead from Ridge (1457 vs 1766) that the global-normalisation LSTM
  had lost; the per-regime gain is now confirmed on both 6-condition
  subsets. Order-domain features (characteristic defect frequencies
  in shaft-speed-relative units) lift CWRU recording-wise diagnosis
  to accuracy **0.802** / F1-macro 0.689 — the best honest CWRU
  result, with InnerRace finally partially readable in-domain
  (109/177). But the CWRU↔Paderborn transfers fail for the *third*
  feature family in a row (Normal F1 0.000 in both directions):
  with hand-crafted features, the cross-rig question is closed
  negatively.

Platform documents:
[`platform/DATASET_REGISTRY.md`](platform/DATASET_REGISTRY.md) ·
[`platform/DATASET_RESEARCH.md`](platform/DATASET_RESEARCH.md) ·
[`platform/EXPERIMENTS_REPORT.md`](platform/EXPERIMENTS_REPORT.md) ·
[`platform/FAULT_DIAGNOSIS.md`](platform/FAULT_DIAGNOSIS.md)

The dashboard gained four views fed from the experiment records —
**Platform**, **Datasets**, **Experiments** and **Research** — alongside
the original Fleet / Machine / Maintenance / Model Info views.

## Tech stack

| Layer | Technology |
|---|---|
| ML lab | Python 3.11, PyTorch, NumPy, Pandas, scikit-learn, Matplotlib, Captum |
| Model exchange | ONNX (opset 17), ONNX Runtime |
| Edge engine | Rust, `ort`, Axum, Tokio, Serde, rusqlite (SQLite) |
| Maintenance | SQLite, config-driven recommendation policy |
| Simulator | Python (FD001 replay) |
| Dashboard | React, TypeScript, Vite, Tailwind CSS, Recharts |
| Testing | pytest, cargo test, end-to-end integration tests |

## Quick start (Windows 11 / PowerShell)

> Full, verified runbook lands with Phase A1. Reference flow:

```powershell
git clone <private-repo-url> PREDICAI
cd PREDICAI

# 1 — Python ML lab
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r ml\requirements.txt
powershell -File ml\scripts\download_data.ps1   # fetches FD001 (see docs/DATASET.md)

# 2 — Train + evaluate + export (after Phase A phases complete)
python -m predicai_ml.train --config ml\configs\fd001_lstm.yaml
python -m predicai_ml.export_onnx

# 3 — Rust edge engine
cargo run --release --manifest-path edge\Cargo.toml
# → GET  http://localhost:8080/health
# → POST http://localhost:8080/predict_rul

# 4 — Dashboard
cd dashboard; npm install; npm run dev
```

**Example API call**

```powershell
$body = @{
  machine_id = "ENGINE-001"
  window     = @(, @(0.12, 0.44) )  # 30 × F normalized-or-raw per API contract
} | ConvertTo-Json -Depth 5
Invoke-RestMethod -Method Post -Uri http://localhost:8080/predict_rul `
  -ContentType "application/json" -Body $body
```

```json
{ "machine_id": "ENGINE-001", "predicted_rul": 42.0, "unit": "cycles", "status": "ok" }
```

*(Example output shape only — `42` is illustrative, not a measured result.)*

## Project structure

```
PREDICAI/
├── plan.md               # phased roadmap + gates (the working contract)
├── docs/                 # research report, architecture, dataset card, devlog
├── ml/                   # Python ML lab (data, model, training, XAI, export)
├── edge/                 # Rust inference API (Axum + ONNX Runtime)
├── maintenance/          # SQLite maintenance history + policy engine
├── simulator/            # FD001 replay → Rust API
├── dashboard/            # React + TypeScript monitoring UI
├── platform/             # Phase 8 multi-dataset platform (registry, adapters, experiments)
└── tests/e2e/            # end-to-end integration tests
```

## Dataset

NASA **C-MAPSS Turbofan Engine Degradation Simulation** (Saxena et al.,
PHM 2008), subset **FD001**: 100 train + 100 test engines, 3 operational
settings, 21 sensors, run-to-failure trajectories. RUL for training engines
is `life − cycle`; test ground truth comes from `RUL_FD001.txt`. Download
instructions and licence notes: [`docs/DATASET.md`](docs/DATASET.md).
The dataset is **not** redistributed in this repository.

## Key engineering decisions

- **Regression, not classification** — the useful question is *how many
  cycles remain*, not a binary fail/no-fail.
- **Piecewise RUL cap (starting at 125)** — the literature convention,
  validated for this model by an ablation (125 vs 150 vs none) in Phase A4.
- **Engine-wise splits + train-only scaler** — no leakage; scores that
  survive scrutiny.
- **One preprocessing artifact** shared by Python and Rust, with numerical
  parity tests (PyTorch ↔ ONNX ↔ Rust) as merge gates.
- **Explainability with honest methods** — permutation importance +
  Integrated Gradients primary; SHAP secondary and labelled experimental.
- **Recommendations are policy, not prophecy** — thresholds live in config
  and every recommendation carries its reasoning.

## Roadmap

- [x] Phase 0 — Research & feasibility (`docs/RESEARCH_REPORT.md`, `plan.md`)
- [ ] Phase A — Python ML lab: data, preprocessing, LSTM, evaluation, XAI, ONNX
- [ ] Phase B — Maintenance history intelligence (SQLite + policy engine)
- [ ] Phase D — Rust edge engine (ONNX Runtime API)
- [ ] Phase E — Sensor simulation (FD001 replay)
- [ ] Phase F — Dashboard
- [ ] Phase G — Integration, benchmarks, demo pack

## Limitations (stated plainly)

- Trained and evaluated on **simulated** benchmark data (FD001, one
  operating condition, one fault mode) — generalization to real machines
  or other subsets (FD002–FD004) is future work, not a claim.
- Predictions are **estimates in cycles**, not guaranteed failure dates.
- Maintenance recommendations follow a configurable policy; they are not
  validated against real maintenance outcomes.
- Demo maintenance records are synthetic.

## Author

**Arvind Kushwaha** — B.Tech, AI & ML, Raj Kumar Goel Institute of
Technology (RKGIT), Ghaziabad.

## Licence

MIT (code). Dataset © NASA — see `docs/DATASET.md` for its terms.

---

<div align="center">
<sub>PREDICAI — Predict Today. Prevent Tomorrow.</sub>
</div>
