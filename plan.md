# PREDICAI — plan.md

> **Predictive Intelligence using Artificial Intelligence**
> Tagline: *Predict Today. Prevent Tomorrow.*
>
> Status: **Phase 0 (Research) complete — awaiting approval to start Phase A1**
> Repo visibility: **PRIVATE** (until the owner explicitly says otherwise)
> Last updated: 13 Nov 2026
> Owner: Arvind Kushwaha

This file is the single source of truth for *kab, kya, kaise*. No phase starts
until the previous phase's gate is passed and verified. No result is written
anywhere (README, report, CV, presentation) until it was actually measured.

---

## 0. Non-negotiable rules (har phase par lagu)

1. **No fabricated results.** RMSE / MAE / NASA score / latency / memory numbers
   appear only after a real run, with the run's config + seed + log saved.
2. **RUL unit is cycles.** Never call cycles "hours" without a documented conversion.
3. **Simulated ≠ real.** Simulator data and synthetic maintenance records are
   always labelled `SIMULATED` / `SYNTHETIC DEMO` in DB, API and UI.
4. **No leakage.** Scaler, feature selection and any fitted statistic are fit on
   training engines only. Windows never mix engine IDs. Splits are engine-wise.
5. **Preprocessing parity.** Python and Rust use one shared artifact
   (`preprocessing.json`) — same features, same order, same scaler values.
   Parity is proven by a test, not by intention.
6. **Small verifiable phases.** Each phase ends with tests + a short completion
   report naming the languages used.
7. **Local-first.** No Kubernetes / Kafka / cloud infra. Windows 11 + PowerShell
   is the reference environment.
8. **Never overwrite blindly.** Inspect existing files before editing; keep
   working implementations; changes go on branches, merges only on owner's word.
9. **Maintenance history is separate.** It is NOT an LSTM input and servicing
   does NOT reset RUL/degradation in the model.
10. **Private repo.** Do not make the GitHub repo public, do not push the NASA
    raw dataset if its licence mirror forbids redistribution — link to the
    official source and ship a download/prep script instead.

---

## 1. Final architecture (target)

```
                    ┌─────────────────────────────────────────────┐
  NASA C-MAPSS      │  PYTHON ML LAB  (ml/)                        │
  FD001 files  ───► │  EDA → Preprocess → RUL labels → Windows    │
                    │  → LSTM train → Evaluate → Explain          │
                    └──────────────┬──────────────────────────────┘
                                   │ export
                                   ▼
                    cmapss_lstm.onnx + preprocessing.json + metadata.json
                                   │
                    ┌──────────────▼──────────────────────────────┐
  Simulator ───────►│  RUST EDGE ENGINE  (edge/)  Axum + ort       │
  (replay FD001     │  validate → normalize → ONNX → RUL (cycles)  │
   test engines)    └──────────────┬──────────────────────────────┘
                                   │ REST JSON
                    ┌──────────────▼──────────────────────────────┐
                    │  DASHBOARD  (dashboard/)  React + TS         │
                    │  Overview · Sensors · RUL · History · XAI    │
                    └──────────────▲──────────────────────────────┘
                                   │
                    ┌──────────────┴──────────────────────────────┐
                    │  MAINTENANCE MODULE  (maintenance/ + SQLite) │
                    │  Machines · Service records · Policy engine  │
                    │  → recommendations with stated reasoning     │
                    └─────────────────────────────────────────────┘
```

---

## 2. Repository structure (improved from the brief, with reasons)

```
PREDICAI/
├── plan.md                  # this roadmap (copied from the approved plan)
├── README.md
├── .gitignore               # data/raw, artifacts, *.db, target/, node_modules
├── docs/
│   ├── RESEARCH_REPORT.md   # Phase 0 report (20-point feasibility)
│   ├── ARCHITECTURE.md      # diagrams + data contracts (Phase A1)
│   ├── DATASET.md           # FD001 card: source, licence, columns, splits
│   └── DEVLOG.md            # dated development log, one entry per work session
├── ml/
│   ├── configs/
│   │   └── fd001_lstm.yaml  # every hyperparameter + seed lives here, not in code
│   ├── data/
│   │   ├── raw/             # gitignored — train_FD001.txt etc. (download script)
│   │   └── processed/       # gitignored — windowed .npz caches
│   ├── src/predicai_ml/
│   │   ├── data.py          # loading + column schema + validation
│   │   ├── preprocessing.py # RUL labels, clipping, scaler, windows
│   │   ├── dataset.py       # PyTorch Dataset / DataLoader builders
│   │   ├── model.py         # LSTM regressor (config-driven)
│   │   ├── losses.py        # MSE + experimental asymmetric loss
│   │   ├── train.py         # loop, early stopping, checkpointing
│   │   ├── evaluate.py      # RMSE / MAE / NASA score + error analysis
│   │   ├── explain.py       # permutation + integrated gradients (+ SHAP subset)
│   │   └── export_onnx.py   # ONNX export + PyTorch↔ONNX parity check
│   ├── scripts/
│   │   ├── download_data.ps1
│   │   ├── run_eda.py
│   │   └── train.py         # thin CLI entry: python -m / script wrapper
│   ├── tests/               # pytest — preprocessing, windows, model shapes
│   └── artifacts/           # gitignored except metadata examples
│       ├── cmapss_lstm.onnx
│       ├── preprocessing.json
│       └── model_metadata.json
├── maintenance/
│   ├── schema.sql           # machines, service_records, recommendations
│   ├── maintenance.py       # SQLite CRUD (Python reference implementation)
│   └── policy.py            # recommendation rules (config thresholds)
├── edge/
│   ├── Cargo.toml           # pinned: ort = "=2.0.0-rc.??" (see Risk R-3)
│   ├── src/
│   │   ├── main.rs          # Axum server bootstrap
│   │   ├── model.rs         # ONNX session (load once, share via Arc)
│   │   ├── preprocess.rs    # reads preprocessing.json — no hardcoded stats
│   │   ├── routes.rs        # GET /health, POST /predict_rul
│   │   ├── types.rs         # request/response DTOs + validation errors
│   │   └── maintenance.rs   # SQLite read/write for history + recommendations
│   └── tests/               # cargo test — validation + inference parity
├── simulator/
│   └── simulator.py         # replays FD001 test engines cycle-by-cycle
├── dashboard/
│   ├── package.json         # React + TypeScript + Vite + Tailwind
│   └── src/                 # pages per §11 of the brief
└── tests/
    └── e2e/                 # full flow: simulator → Rust API → stored history
```

**Changes vs. the brief's sketch, and why:**
- `ml/src/predicai_ml/` package instead of loose folders — importable, testable,
  and the same code is reused by training, export and the simulator.
- One `preprocessing.json` artifact as the Python↔Rust contract — prevents the
  #1 production bug in ML systems (train/serve skew).
- `configs/*.yaml` — experiments are reproducible and diff-able in git.
- Maintenance starts as a Python module + shared `schema.sql`; the Rust engine
  reads/writes the same SQLite file. No duplicate schema definitions.

---

## 3. Phases, tasks and gates

### Phase 0 — Research & Feasibility ✅ DONE (13 Nov 2026)
- [x] Dataset research (FD001 structure, RUL rules, official test format)
- [x] RUL clipping 125 — literature check (convention, not proven optimum → ablation planned)
- [x] LSTM architecture recommendation + training methodology
- [x] ONNX + Rust `ort` feasibility, including the RC-version risk
- [x] Explainability strategy correction (SHAP limits on LSTM sequences)
- [x] `plan.md`, `RESEARCH_REPORT.md`, `README.md` drafted
- **Gate:** Owner approval of the research report. ⏳ **Waiting.**

### Phase A1 — Project scaffold + Dataset foundation
Languages: Python, PowerShell, Markdown
- [ ] Create private GitHub repo `PREDICAI`, `.gitignore`, folder structure above
- [ ] Python venv + pinned `requirements.txt` (versions in §6)
- [ ] `download_data.ps1` — fetch FD001 from the official NASA / Kaggle mirror,
      verify file list + row counts; document source + licence in `docs/DATASET.md`
- [ ] `data.py` — loader with the exact 26-column schema; schema validation
- [ ] EDA notebook/script output saved as figures + a written EDA summary:
      per-sensor stats, missing values (expected: none — verify, don't assume),
      constant / near-constant sensors, engine life distribution
- **Tests:** loader shape test; column-name test; row-count test vs known FD001
  totals (train 20,631 rows / 100 engines; test 13,096 rows / 100 engines)
- **Gate:** EDA summary reviewed; constant-sensor list is *measured*, not copied
  from a blog. Feature set decision recorded in `docs/DATASET.md`.

### Phase A2 — Preprocessing pipeline
Languages: Python
- [ ] RUL label generation — train: `max_cycle(engine) − cycle`;
      test: `RUL_FD001.txt value + (max_cycle − cycle)` per cycle
- [ ] RUL clipping (piecewise-linear cap) — config value, default 125
- [ ] Engine-wise split: train engines → train/val by engine ID (e.g. 80/20,
      seed fixed); official test set stays untouched until evaluation
- [ ] Scaler fit on train split only; save to `preprocessing.json`
      (feature names + order, mean, scale, seq_len, clip value)
- [ ] Sliding windows: length 30, stride 1, per engine, label = RUL at the
      window's last cycle; short engines (<30 cycles) handled by left-padding
      rule documented + tested (matters for the test set's last-window rule)
- **Tests:** RUL calculation on a hand-made toy engine; window shape
  `(N, 30, F)`; no window mixes engines; scaler statistics equal train-only
  statistics; val/test transform uses saved scaler unchanged
- **Gate:** All preprocessing tests green; one sample window traced by hand
  from raw row → normalized tensor and matched.

### Phase A3 — LSTM model + Training
Languages: Python (PyTorch)
- [ ] `model.py` — config-driven LSTM regressor (recommended start in the
      research report §7: 2-layer LSTM, hidden 128 → 64 path via head, dropout,
      linear head to 1 output)
- [ ] Training loop: Adam, MSE loss, batch 64, grad-clip, cosine/plateau LR,
      early stopping on val RMSE (patience ~10), best checkpoint saved
- [ ] Seed control (python / numpy / torch) + config snapshot saved per run
- [ ] Curves: train/val loss + val RMSE plots saved per run
- **Tests:** forward shape `(B,30,F) → (B,1)`; overfit-one-batch sanity test;
  checkpoint reload reproduces predictions
- **Gate:** A full training run completes end-to-end and produces a checkpoint
  + curves. **No performance claim yet** — that is A4.

### Phase A4 — Evaluation + Error analysis
Languages: Python
- [ ] Official-style test evaluation: **last window per test engine** →
      RMSE, MAE, NASA PHM08 score; also val-set metrics for all windows
- [ ] Prediction-vs-actual plot; error by true-RUL band (0–25 / 25–50 / 50–125);
      per-engine worst cases listed
- [ ] Baseline comparison (same pipeline): mean-predictor + Ridge on flattened
      windows — the LSTM must beat these to justify itself
- [ ] Clipping ablation: cap 125 vs 150 vs no-cap (train 3 short runs, same seed)
      → pick the cap on val RMSE, record the decision + numbers
- **Gate:** Metrics table exists **from real runs only**, with config + seed;
  README results section may now be filled — with these numbers and no others.

### Phase A5 — Explainability
Languages: Python
- [ ] Primary: permutation importance per sensor (val set, RMSE increase)
- [ ] Primary: Integrated Gradients (Captum) on representative windows
      (low / mid / high true RUL) → per-sensor attribution plots
- [ ] Secondary (labelled experimental): SHAP on a flattened-window surrogate
      or a small background sample — with its limits written next to the chart
- [ ] Human-readable summary: "which sensors move the prediction, in which
      direction" — association language only, never causal claims
- **Gate:** At least two methods agree on the top sensors, or the disagreement
  is documented honestly.

### Phase A6 — ONNX export (Phase C of the brief)
Languages: Python
- [ ] Export best checkpoint → `ml/artifacts/cmapss_lstm.onnx`
      (opset 17, dynamic batch axis, `model.eval()`, dummy `(1, 30, F)`)
- [ ] `onnx.checker` + ONNX Runtime (Python) load test
- [ ] **Parity test:** PyTorch vs ONNX outputs on ≥100 fixed windows,
      max abs diff tolerance `1e-4` — must pass before Phase D starts
- [ ] Freeze artifacts: `cmapss_lstm.onnx`, `preprocessing.json`,
      `model_metadata.json` (arch, F, seq_len, clip, metrics, git commit, date)
- **Gate:** Parity test green and artifacts frozen + tagged in git.

### Phase B — Maintenance History Intelligence (can run parallel to A4–A6)
Languages: Python, SQL (SQLite)
- [ ] `schema.sql`: `machines`, `service_records`, `recommendations`
- [ ] CRUD: register machine, add/list service records, cycles-since-service
      computed from a supplied current cycle (never invented)
- [ ] Seed script creates **clearly-labelled synthetic demo records only**
      (`is_synthetic = 1`), e.g. for ENGINE-001 — never presented as real history
- [ ] Policy engine (`policy.py`): inputs = predicted RUL + cycles since
      service + last service type; thresholds live in config with the explicit
      note "policy choices, not validated failure boundaries"; every
      recommendation carries a `reason` string
      (e.g. `RUL 38 cycles (≤ 50 review band) and 320 cycles since last service`)
- **Tests:** CRUD round-trip; cycles-since-service arithmetic; policy outputs
  for hand-computed cases; synthetic flag present on all seed rows
- **Gate:** Demo flow shows a machine, its synthetic history, and a reasoned
  recommendation — with the synthetic label visible.

### Phase D — Rust Edge Engine
Languages: Rust
- [ ] `Cargo.toml` pins (exact `ort` RC version recorded; see Risk R-3)
- [ ] Startup: load ONNX once, build session, load `preprocessing.json`;
      fail fast with a clear error if artifacts are missing/mismatched
- [ ] `GET /health` → `{ status, model_loaded, seq_len, num_features }`
- [ ] `POST /predict_rul` — body: `machine_id`, `window: [[f32; F]; 30]`
      → validate shape/order → normalize → tensor `(1, 30, F)` → inference
      → `{ machine_id, predicted_rul, unit: "cycles", status }`
      Errors: structured JSON (wrong seq len, wrong feature count, NaN, …)
- [ ] Prediction log table (SQLite): timestamp, machine, RUL, model version
- [ ] Maintenance endpoints (read history + recommendation) over the same DB
- **Tests (cargo test):** health; validation rejections; **Rust vs Python
  parity** on the same fixed windows (tolerance `1e-3` end-to-end);
  inference smoke test against the frozen ONNX file
- **Benchmark (measured, not claimed):** latency p50/p95 over 1,000 requests
  locally; binary + memory footprint noted in DEVLOG
- **Gate:** Parity + API tests green; a `curl`/PowerShell call returns a sane
  RUL for a real FD001 test window.

### Phase E — Live Sensor Simulation
Languages: Python
- [ ] Replay mode (primary): stream a chosen FD001 **test** engine cycle by
      cycle, maintain the rolling 30-window, call the Rust API each cycle,
      store predictions — all labelled `SIMULATED REPLAY (NASA FD001 test data)`
- [ ] Synthetic mode (secondary, optional): perturbed copies for demo variety,
      labelled `SYNTHETIC`
- [ ] Prediction history persists (SQLite) for the dashboard charts
- **Gate:** One full engine replay runs start→end with a prediction per cycle
  and no shape/validation errors.

### Phase F — Dashboard
Languages: TypeScript, React, Tailwind CSS
- [ ] Vite + React + TS scaffold; PREDICAI industrial identity
      (dark slate + amber/cyan accents, no gradients-as-decoration, no fake stats)
- [ ] Pages: Overview · Machine monitoring · Sensor trends (Recharts) ·
      RUL gauge + prediction history · Maintenance history · Recommendations ·
      Explainability (A5 charts) · Model info (real A4 metrics + metadata)
- [ ] Every panel shows its data source label: `FD001 replay` / `SYNTHETIC DEMO`
- [ ] States: loading, API-down, empty history — designed, not afterthoughts
- **Gate:** Dashboard runs against the live Rust API + simulator and shows one
  engine end-to-end; screenshots taken for README/report from the real app.

### Phase G — Integration, hardening, presentation
Languages: all
- [ ] E2E test: simulator → API → DB → dashboard data endpoints
- [ ] Full test sweep: pytest + cargo test + dashboard build, all green
- [ ] README finalized (real metrics, real screenshots, real benchmark numbers)
- [ ] `docs/DEVLOG.md` complete; demo script (5-minute walkthrough) written
- [ ] Viva/interview notes: why LSTM, why clipping, why ONNX, why Rust,
      leakage answers — one page, in the owner's own understandable words
- **Gate:** Fresh-clone runbook (PowerShell, §6) executed once from scratch.

### Stretch (only after the MVP gate, and only on owner's word)
- FD003 (2 fault modes) → FD002/FD004 with regime-wise normalization
- 1D-CNN / CNN-LSTM comparison under the identical pipeline
- MC-Dropout uncertainty intervals; asymmetric loss experiment (from A3 config)
- INT8 dynamic quantization of the ONNX model + latency comparison
- Raspberry Pi deployment note (documented, hardware not required)

---

## 4. MVP definition (what "done" means first)

A demonstrable, honest MVP = **A1–A6 + B + D + E + a read-only-ish F**:

1. Trained LSTM on FD001 with real RMSE / MAE / NASA score.
2. ONNX parity proven Python ↔ ONNX ↔ Rust.
3. Rust API serving predictions on `localhost`.
4. Simulator replaying a real test engine into the API.
5. Dashboard showing sensors, RUL, history, recommendations, XAI.
6. Synthetic maintenance demo clearly labelled.
7. Tests green at every layer.

Anything beyond this is stretch (§3, Phase G list). No stretch work starts
before the MVP gate passes.

---

## 5. Key decisions (locked for the MVP, revisit only with evidence)

| # | Decision | Choice | Reason (detail in research report) |
|---|----------|--------|-------------------------------------|
| D-1 | Task | RUL **regression**, not classification | The brief's core question is "how many cycles left" |
| D-2 | Subset | FD001 only | 1 condition, 1 fault mode — the correct starting point |
| D-3 | Window | 30 cycles, stride 1, label at last step | Literature standard; matches brief |
| D-4 | RUL cap | Start 125, **ablate** 150 / none in A4 | 125 is a convention, not an optimum |
| D-5 | Features | Measured in A1 EDA; expect ~14 sensors (7 constant dropped) + decide settings columns by variance | Never hardcode someone else's feature list |
| D-6 | Normalization | StandardScaler (z-score), fit on train engines only | Simple, exactly reproducible in Rust |
| D-7 | Split | Engine-wise 80/20 train/val + official test | Random window splits leak and inflate scores |
| D-8 | Test metric | Last window per test engine | Matches the official/PHM08 comparison protocol |
| D-9 | Loss | MSE first; asymmetric loss = flagged experiment | NASA score already penalizes late predictions in evaluation |
| D-10 | Explainability | Permutation + Integrated Gradients primary; SHAP secondary/experimental | SHAP on raw LSTM sequences is methodologically shaky |
| D-11 | ONNX | Opset 17, dynamic batch, seq fixed at 30 | Safe modern default; seq fixed simplifies the Rust contract |
| D-12 | Rust runtime | `ort` crate, **exact RC version pinned** | `ort` 2.0 is still RC — pinning is mandatory (Risk R-3) |
| D-13 | Maintenance data | Synthetic, flagged, separate DB concern | FD001 has no service records — inventing "real" ones is fabrication |
| D-14 | Thresholds | Policy config, labelled as policy | No threshold is a scientifically validated failure boundary |

---

## 6. Environment + version pins (Windows 11 / PowerShell)

Reference commands (full runbook is written in Phase A1 and kept in README):

```powershell
# Python (3.11 recommended — widest wheel support for torch + shap + captum)
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r ml\requirements.txt

# Rust (stable via rustup)
cargo --version
cargo build --release --manifest-path edge\Cargo.toml

# Dashboard (Node LTS 20+)
cd dashboard; npm install; npm run dev
```

Planned pins (finalized + frozen in A1 after a real install test):
- Python 3.11 · PyTorch 2.x (CPU first; CUDA optional) · NumPy · Pandas ·
  scikit-learn · Matplotlib · onnx + onnxruntime (parity checks) ·
  captum (IG) · shap (secondary) · pyyaml · pytest
- Rust stable · `ort` (exact `=2.0.0-rc.*`) · axum · tokio · serde ·
  serde_json · rusqlite (bundled SQLite) · thiserror / anyhow
- Node 20 LTS · React 18/19 · TypeScript · Vite · Tailwind · Recharts

---

## 7. Risks (top — full list in the research report §17)

| ID | Risk | Mitigation in this plan |
|----|------|--------------------------|
| R-1 | Train/serve preprocessing skew (Python vs Rust) | Single `preprocessing.json` + parity tests in A6 and D |
| R-2 | Data leakage inflating scores | Engine-wise splits, train-only scaler, leakage tests in A2 |
| R-3 | `ort` crate has **no stable 2.0 release** (RC since 2024) | Pin exact RC, isolate model code in `model.rs`, fallback = `tract` (pure Rust) if a blocker hits |
| R-4 | SHAP misread as causal truth | Two primary methods + association-only language (A5) |
| R-5 | Short test engines (< 30 cycles) break the last-window rule | Documented padding rule + test in A2 |
| R-6 | Thresholds mistaken for science | Policy labelling everywhere (D-14) |
| R-7 | Scope creep (FD002–004, Pi, quantization early) | Stretch list is gated behind the MVP |
| R-8 | Dataset redistribution / licence | Download script + link, raw data gitignored |

---

## 8. Working agreement

- Work phase by phase; after each phase: tests + short report (what was built,
  test evidence, languages used) → owner reviews → next phase.
- README results/benchmark sections stay as `—` placeholders until A4/D
  produce real numbers.
- Repo stays **private**; visibility change only on the owner's explicit word.
- This plan changes only by editing this file with a dated note in DEVLOG.

*Phase 0 deliverables: this plan + `RESEARCH_REPORT.md` + `README.md` draft.*
*Next action: owner approval → Phase A1 scaffold.*
