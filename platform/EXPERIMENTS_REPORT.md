# PREDICAI Platform — Experiments Report (Phase 2 design + Phase 4, wave 1)

**Date:** 2026-11-13 · **Code:** `platform/platform_lib/` · **Records:** `platform/experiments/` (10 records + `INDEX.json`)
**Rule honoured throughout:** every number below is copied from an experiment record
produced by an actual run in this session. Nothing is estimated, and the existing
FD001 system (`ml/`, `edge/`, `dashboard/`) was not modified (verified: `git status`
shows new files under `platform/` only).

Environment for all runs: Python 3.12.3, scikit-learn 1.5.2, numpy 1.26.4,
pandas 2.1.4, torch 2.15.0+cpu, xgboost 3.4.1, lightgbm 4.7.0 (the last two were
installed into the project `.venv` during this phase), seed 42 everywhere,
local git commit `8d3f931d488c881ded36109ed526534d22aa48ad`.

---

## 1. AI4I 2020 — Task A, failure classification

Protocol: stratified 80/20 (train 8,000 / test 2,000; 68 failures in test).
Inputs: 5 process features + one-hot `Type`. Excluded: `UDI`, `Product ID`
(identifiers) and TWF/HDF/PWF/OSF/RNF (the failure-mode flags *define* the
target — using them would be leakage). All-negative baseline accuracy: **0.9660**
— every model's accuracy must be read against that number, which is why F1 /
recall are the headline metrics.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Confusion [[TN,FP],[FN,TP]] | Train (s) | Infer (ms/row) |
|---|---|---|---|---|---|---|---|---|
| Logistic Regression (balanced) | 0.8250 | 0.1421 | 0.8235 | 0.2424 | 0.9069 | [[1594, 338], [12, 56]] | 0.02 | 0.0004 |
| Random Forest (balanced) | 0.9800 | 0.9118 | 0.4559 | 0.6078 | 0.9636 | [[1929, 3], [37, 31]] | 1.91 | 0.0317 |
| Hist Gradient Boosting (balanced sample weights) | 0.9790 | 0.6625 | 0.7794 | 0.7162 | 0.9691 | [[1905, 27], [15, 53]] | 0.87 | 0.0256 |
| XGBoost (scale_pos_weight ≈ 28.5) | 0.9830 | 0.7361 | 0.7794 | 0.7571 | 0.9709 | [[1913, 19], [15, 53]] | 0.89 | 0.0052 |
| **LightGBM (balanced)** | **0.9855** | 0.8000 | 0.7647 | **0.7820** | 0.9703 | [[1919, 13], [16, 52]] | 0.82 | 0.0255 |

**XAI record** — permutation importance on the Random Forest (test split,
scoring = F1, 10 repeats): Torque [Nm] **0.452**, Rotational speed [rpm] **0.428**,
Air temperature [K] 0.199, Process temperature [K] 0.154, Tool wear [min] 0.143,
Type_L 0.031, Type_H 0.013, Type_M 0.005. Torque and speed dominate — consistent
with AI4I's power-failure rule (power ∝ torque × speed) and heat-dissipation rule,
which is the expected sanity check for this dataset.

**Limitation (binding):** AI4I is *synthetic* and its failures are deterministic
rules of the input features, so these scores measure how well models recover the
generating rules — near-separability by construction. They are **not** evidence
of performance on real machines, and must never be quoted as such.

## 2. SECOM — Tasks A (classification) + C (anomaly detection)

Protocol: stratified 80/20 (train 1,253 / test 314; 21 fails in test).
Preprocessing fitted on the train split only: 590 raw features → **566** after
dropping columns >50 % missing → **450 surviving** after dropping constants;
median imputation (train medians); StandardScaler (train). Label map:
raw −1 (pass) → 0, raw +1 (fail) → 1. All-negative baseline accuracy: 0.9331.

| Model | Task | Accuracy | Precision | Recall | F1 | ROC-AUC | Confusion [[TN,FP],[FN,TP]] | Train (s) |
|---|---|---|---|---|---|---|---|---|
| Logistic Regression (balanced) | A | 0.8408 | 0.1081 | 0.1905 | 0.1379 | 0.6203 | [[260, 33], [17, 4]] | 0.04 |
| Random Forest (balanced) | A | 0.9331 | 0.0000 | 0.0000 | 0.0000 | 0.8019 | [[293, 0], [21, 0]] | 3.92 |
| **Isolation Forest** (fit on pass rows only, contamination 0.0664) | C | 0.9013 | 0.2500 | 0.2381 | **0.2439** | — | [[278, 15], [16, 5]] | 0.39 |

False-alarm rate (Isolation Forest): **0.0512** (15 of 293 pass rows flagged).

Honest reading: SECOM is a genuinely weak-signal dataset. The Random Forest
*ranks* failures reasonably (ROC-AUC 0.80) yet predicts zero failures at the
default 0.5 threshold — a threshold/imbalance artefact, reported as measured
rather than tuned away. The unsupervised Isolation Forest, which never sees a
fail label in fitting, beats both supervised models on F1 — with only 104 fails
in the whole dataset (83 in train), that is a plausible and defensible result,
and it is exactly why SECOM carries both a Task A and a Task C entry. All
estimates rest on 21 test failures, so they are high-variance; treat them as
order-of-magnitude, not precise rankings.

## 3. C-MAPSS FD003 — Task B, RUL regression

Protocol: engine-wise 80/20 train/val (80/20 engines, seed 42) + the official
provided test set scored as **last window per engine** (n = 100). RUL cap 125,
window 30, z-score scaler fit on train engines only.

**Feature derivation (from FD003's own train split, std < 0.005 dropped):**
15 features kept — sensors 2, 3, 4, **6**, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21.
Dropped: setting_1 (std 0.0022), setting_2 (0.0003), setting_3 (0.0000),
sensor_1, sensor_5, sensor_16, sensor_18, sensor_19 (all exactly constant),
sensor_10 (0.0035).

Two findings worth stating plainly:

1. **The settings were candidates and were dropped by the rule, not by hand.**
   FD003 has a *single* operating condition (the six-condition subsets are
   FD002/FD004 — an earlier brief misstated this), so its settings are constant
   setpoints carrying no signal; setting_3 would additionally break z-scoring
   (zero variance). The same rule will keep the settings automatically on
   FD002/FD004, where they vary across regimes.
2. **Sensor_6 is kept for FD003** (train std ≈ 0.018) although the FD001
   pipeline drops it (FD001 std 0.00139). Per-subset derivation, not inherited
   lists, is what catches this — plausibly related to FD003's second (fan)
   fault mode.

| Model | Test RMSE | Test MAE | Test NASA | Val RMSE (all windows) | Train time | Infer (ms/window) |
|---|---|---|---|---|---|---|
| Ridge (flattened 30×15 windows, α = 1.0) | 16.358 | 13.832 | 476.9 | 15.254 | 0.11 s | 0.0019 |
| **LSTM (same architecture/recipe as shipped FD001 model; local CPU)** | **15.043** | **10.512** | **451.8** | 11.018 | 475.2 s (28 epochs, early stop; best val RMSE 11.018) | 0.2054 |

The LSTM beats the Ridge baseline on all three test metrics. Context only (not
a comparison — different subset, different fault modes, and cross-dataset score
ranking is prohibited by the platform's own model-selection rules): the shipped
FD001 model scored RMSE 13.73 / MAE 9.89 / NASA 431.3 in the owner's Kaggle GPU
run. FD003's extra fault mode makes it the harder problem, and the local CPU
run lands in a sensible band relative to both that reference and FD003
literature.

## 4. Files created

```
platform/ARCHITECTURE.md                  Phase 2 design
platform/EXPERIMENTS_REPORT.md            this report
platform/platform_lib/__init__.py
platform/platform_lib/adapters/base.py        DatasetAdapter interface
platform/platform_lib/adapters/ai4i.py        AI4I 2020 (verified: 10,000 x 14, 339 failures)
platform/platform_lib/adapters/secom.py       SECOM (verified: 1,567 x 590, labels -1/+1 -> 0/1)
platform/platform_lib/adapters/cmapss.py      subset-parameterised C-MAPSS wrapper over predicai_ml.data
platform/platform_lib/rul_pipeline.py         per-subset feature derivation + windowing
platform/platform_lib/benchmarking/run_benchmark.py   benchmarking engine + all 10 experiments
platform/experiments/*.json               10 experiment records + INDEX.json
platform/tests/conftest.py, test_adapters.py, test_experiment_schema.py
platform/data/raw/{ai4i,secom,cmapss}/    downloaded raw data (see note below)
```

Run commands (from `platform/`): `../.venv/bin/python -m platform_lib.benchmarking.run_benchmark --set ai4i|secom|fd003|all`

**Tests: 16/16 passed** (adapter shape/label/feature-derivation checks for all
three datasets, per-record schema checks for all 10 experiment records, index
consistency). Run: `../.venv/bin/python -m pytest tests/ -q`

Note on raw data: `platform/data/raw/` (~28 MB incl. zips and the nested
C-MAPSS archive) is currently **not** covered by the repo `.gitignore`
(the existing rules cover `ml/data/raw` only). Recommendation: add
`platform/data/` to `.gitignore` before any commit — the platform does not
need to redistribute datasets (AI4I/SECOM are CC BY 4.0; C-MAPSS is NASA
research-use with citation).

## 5. What remains for later phases

- **CWRU adapter (wave 2):** `.mat` vibration recordings; window slicing with
  splits by recording/condition (never by window); Task A as *diagnosis*
  wording. The adapter interface already accommodates it.
- **FD002 / FD004:** the derivation rule keeps their 3 settings automatically;
  per-regime normalisation is the open design question to settle before runs.
- **Cross-dataset generalisation (Phase 6) — designed, not run:** train the
  LSTM on FD001 and score on FD003. Feature alignment must be decided first:
  the FD001 model consumes 14 sensors (no sensor_6) while the FD003 derivation
  keeps 15, so the transfer run should train on the 14-sensor intersection,
  and the scaler question (source-fitted vs target-fitted statistics) must be
  stated as part of the protocol. Degradation vs the native FD003 numbers
  above is the expected, reportable result — FD003's fan fault mode is unseen
  in FD001 training.
- **SHAP (Phase 7):** TreeExplainer summaries for the AI4I LightGBM/XGBoost and
  SECOM models, stored in the records' `extra` block alongside the existing
  permutation-importance record.
- **Dashboard / API integration (Phase 8):** the dashboard should render
  benchmark tables directly from `experiments/INDEX.json` + records, and the
  model-selection layer (ARCHITECTURE.md §5) gates any serving: per
  (dataset, task) ranking only, preprocessing artefact versioned with the model.
- **Maintenance-history experiment:** still *not run — no suitable data*. None
  of the wave-1 datasets contains real service logs; the channel stays
  structurally absent rather than synthesised (ARCHITECTURE.md §7).
