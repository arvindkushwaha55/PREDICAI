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

---

# Wave 2 — Phases 4/6/7 continuation (2026-11-15)

**Records:** 7 new (`platform/experiments/`, 17 total + `INDEX.json`) ·
**Code:** `platform/platform_lib/benchmarking/run_wave2.py`,
`platform/platform_lib/adapters/cwru.py` · Same rule as wave 1: every
number below is copied from an experiment record produced by an actual
run in this session; the existing FD001 system was not modified.
Environment: same pinned stack as wave 1 (numpy 1.26.4, pandas 2.1.4,
torch 2.15.0+cpu, scikit-learn 1.5.2, lightgbm 4.7.0) plus shap 0.54.1,
installed this session. Note: shap 0.54.1 declares `numpy>=2`; the
platform keeps numpy 1.26.4 for consistency with the shipped system's
artifacts, and shap's TreeExplainer was smoke-tested and then verified
in the real run below (its retrained-model metrics reproduce the wave-1
LightGBM record exactly, and the ranking is physically sensible).

## 6. C-MAPSS FD002 — Task B, RUL regression

Protocol: identical to FD003 wave 1 (engine-wise 80/20, seed 42; official
test set, last window per engine, **n = 253** for FD002; cap 125,
window 30, scaler fit on train engines only).

**Feature derivation (FD002's own train split, std < 0.005 dropped):**
23 features kept — all 3 settings plus 20 sensors. Dropped: sensor_16
only. This is the six-condition counterpart of the FD003 finding: where
the operating condition varies, the settings *and* the regime-dependent
sensors (1, 5, 10, 18, 19 — constant within FD001/FD003's single regime)
carry real signal and the rule keeps them automatically.

| Model | Test RMSE | Test MAE | Test NASA | Val RMSE (all windows) | Train time |
|---|---|---|---|---|---|
| Ridge (flattened 30×23 windows, α = 1.0) | 16.582 | 13.268 | 1267.2 | 17.006 | 0.8 s |
| **LSTM (shipped recipe, local CPU)** | **16.236** | **11.933** | **1099.8** | 17.042 (best, epoch 10; early stop @ 20) | 931 s |

Honest reading: on FD002 the LSTM's margin over Ridge is the smallest
seen so far — 2 % on RMSE (though 10 % on MAE and 13 % on NASA). Its
validation RMSE bottomed at epoch 10 and then *rose* (17.0 → 21.8 by
epoch 20, where early stopping fired); the saved model is the epoch-10
state, so the reported test numbers are that state's. FD002's six
operating regimes make the raw-window problem harder for both models,
and the shipped recipe — tuned on single-condition FD001 — is not
regime-aware (no per-regime normalisation; that remains the open design
question flagged in wave 1).

## 7. C-MAPSS FD004 — Task B, RUL regression

Protocol as FD002 (official test n = 237). Feature derivation produced
the same 23 features as FD002 (all settings kept; sensor_16 dropped).

| Model | Test RMSE | Test MAE | Test NASA | Val RMSE (all windows) | Train time |
|---|---|---|---|---|---|
| Ridge (flattened 30×23 windows, α = 1.0) | 19.425 | 15.708 | 1779.1 | 19.338 | 0.6 s |
| **LSTM (shipped recipe, local CPU)** | **18.809** | **13.538** | 2141.5 | 18.346 (best, epoch 9; early stop @ 19) | 742.6 s |

FD004 (six conditions × two fault modes) is the hardest subset for the
linear baseline, as expected. The LSTM wins on RMSE (−3 %) and MAE
(−14 %) but **loses on the NASA score** (2141.5 vs Ridge's 1779.1):
NASA's asymmetric penalty punishes the LSTM's occasional large
over-predictions more than it rewards its better average accuracy —
the same pattern seen in wave 1 on FD001 (Ridge NASA 407.8 vs shipped
LSTM 431.3), now amplified on the hardest subset. Reported as measured;
on a NASA-score criterion the FD004 ranking would flip, and the
platform's per-(dataset, task) records keep all three metrics precisely
so that choice stays visible rather than hidden behind one number.

## 12. What wave 2 closes, and what remains

Closed from the wave-1 "remains" list: FD002 + FD004 baselines and
LSTMs (§6–7), the FD001→FD003 cross-dataset transfer (§8 — run, with a
negative and therefore informative result), SHAP for AI4I (§9), and the
CWRU adapter (§10 — integrated, with its evaluation limitation stated).
Still open, unchanged from wave 1: per-regime normalisation for the
six-condition subsets (now the clearest modelling gap — both FD002 and
FD004 LSTMs peaked early and then degraded on validation), a
recording-wise CWRU evaluation with more recordings per class, SHAP
for the SECOM models, platform dashboard/API integration (Phase 8),
and the maintenance-history experiment (still no dataset with real
service logs — the channel remains absent, not synthesised).

## 8. Phase 6 — Cross-dataset transfer: FD001 → FD003 (zero-shot)

Protocol (spelled out in the record): the **shipped** FD001 LSTM
(`ml/artifacts/best_model.pt`, owner's Kaggle GPU run) applied unchanged
to FD003's official test set, last window per engine (n = 100). Inputs
built with FD001's exact 14-sensor feature set and FD001's scaler
statistics from `ml/artifacts/preprocessing.json` — no FD003 fitting of
anything: no fine-tuning, no target-domain scaler, no threshold tuning.

| Model | Test RMSE | Test MAE | Test NASA |
|---|---|---|---|
| FD001 LSTM on FD001 (in-domain reference) | 13.729 | 9.886 | 431.3 |
| FD003 LSTM on FD003 (in-domain reference) | 15.043 | 10.512 | 451.8 |
| **FD001 LSTM → FD003 (zero-shot transfer)** | **48.800** | **30.673** | **956,059** |

The degradation is severe: RMSE ×3.2 vs the FD003 in-domain model
(+224 %), and the NASA score — which penalises over-prediction
exponentially — blows up by three orders of magnitude. The per-engine
breakdown (in the record) shows *how* it fails, and it is not uniform:
the median absolute error is 11.5 cycles (half the engines transfer
acceptably), but 24 of 100 engines are off by more than 50 cycles, and
the mean prediction (101.4) sits far above the mean truth (73.8) — the
FD001 model systematically reads many FD003 engines as much healthier
than they are. That is the dangerous direction for a maintenance
system, and it is consistent with the known causes: (1) FD003's second
(fan) fault mode produces degradation signatures FD001 training never
contains, in sensor channels the FD001 model maps to high RUL; (2)
plain distribution shift — FD003 sensor marginals differ from FD001's,
and the source-fitted scaler cannot correct that. Conclusion the
platform is allowed to draw: **zero-shot transfer FD001 → FD003 is not
deployable**; a target-domain model (or at minimum target-domain
adaptation) is required. The reverse direction (FD003 → FD001) was not
run: the wave-1 FD003 training did not save a checkpoint, and
retraining solely to enable it was judged out of scope — stated here
rather than silently omitted.

## 9. Phase 7 — SHAP for AI4I LightGBM

The wave-1 LightGBM (n_estimators 300, class_weight balanced, seed 42,
identical stratified split) was retrained and explained with
`shap.TreeExplainer` over all 2,000 test rows. The retrained model's
metrics reproduce the wave-1 record **exactly** (accuracy 0.9855,
precision 0.8000, recall 0.7647, F1 0.7820, ROC-AUC 0.9703), so the
attributions describe the same model the platform already reports.

Mean |SHAP| ranking (failure class, log-odds scale):
Torque [Nm] **1.362**, Tool wear [min] 1.260, Air temperature [K] 1.144,
Rotational speed [rpm] 0.943, Process temperature [K] 0.905,
Type_M 0.220, Type_H 0.218, Type_L 0.099.

**Comparison with wave-1 permutation importance** (Random Forest,
test-F1): the top-5 *sets are identical* — {Torque, Rotational speed,
Air temperature, Process temperature, Tool wear} — and both methods put
Torque first. The order inside the top group differs: SHAP ranks Tool
wear #2 (permutation: #5) and Rotational speed #4 (permutation: #2).
Across all 8 features the rankings correlate at Spearman ρ = 0.74
(p = 0.037). Plain reading: the two methods, on two different models,
agree on *which* features drive AI4I failure predictions and on the
single dominant one; they disagree on mid-group ordering, which is
expected — permutation importance measures F1 lost when a feature is
destroyed (on the RF), SHAP measures mean attribution magnitude (on
the LightGBM), and correlated process variables (the two temperatures;
torque × speed) split credit differently under the two definitions.
The AI4I synthetic-data caveat from §1 applies unchanged.

## 10. CWRU Bearing — Task A, fault diagnosis (bounded integration)

The official CWRU Bearing Data Center downloads worked. Four 12 kHz
drive-end recordings were fetched from the official file host
(engineering.case.edu), one per class, all 0 HP / 0.007″ faults:
97.mat (Normal, 243,938 samples), 105.mat (IR007_0, 121,265),
118.mat (B007_0, 122,571), 130.mat (OR007@6_0, 121,991). Each file's
identity is self-verified by its internal `X<id>_DE_time` variable and
its size matches the Data Center catalogue. New adapter:
`platform_lib/adapters/cwru.py` — non-overlapping 2048-sample windows →
10 time-domain statistics per window → 296 windows (119 Normal,
59 per fault class). Random Forest (300 trees, balanced), stratified
80/20 split over windows, seed 42:

| Model | Accuracy | F1 (macro) | Per-class F1 | Confusion |
|---|---|---|---|---|
| Random Forest | 1.000 | 1.000 | 1.000 / 1.000 / 1.000 / 1.000 | diagonal (24/12/12/12) |

**This perfect score must be read with its binding limitation:** the
split is over *windows*, and there is only one recording per class, so
windows of the same recording sit in both train and test — the score
shows the four conditions are cleanly separable within these
recordings (and that the adapter + pipeline work end-to-end on real
vibration data), **not** that the classifier generalises to unseen
bearings, loads, or fault sizes. A recording-wise evaluation needs
more recordings per class (the Data Center publishes 4 loads × several
fault sizes) and is the honest next step before any CWRU number is
quoted as diagnosis performance.

## 11. Wave-2 files

```
platform/platform_lib/benchmarking/run_wave2.py   FD002/FD004, transfer, SHAP, CWRU runners
platform/platform_lib/adapters/cwru.py            CWRU adapter (official .mat downloads)
platform/experiments/                             7 new records (17 total) + INDEX.json
platform/tests/test_wave2.py                      wave-2 tests
platform/data/raw/cwru/{97,105,118,130}.mat       official CWRU downloads (gitignored with platform/data/)
```

Run commands (from `platform/`):
`../.venv/bin/python -m platform_lib.benchmarking.run_wave2 --set fd002|fd004|transfer|shap|cwru`
