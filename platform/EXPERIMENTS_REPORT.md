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

---

# Wave 4 — Model expansion & leakage-safe evaluation (2026-11-15)

**Records:** 12 new (29 total in `platform/experiments/`). **Code:**
`platform/platform_lib/benchmarking/run_wave4.py`; **tests:**
`platform/tests/test_wave4.py` — full platform suite **64/64 passed**.
Seed 42 everywhere; the existing FD001 system and all earlier records
are untouched.

Wave 4's theme: broaden the model zoo where waves 1–2 used one family
per task, and replace the two weakest split protocols in the earlier
records with leakage-safe ones (CWRU recording-wise, Paderborn
measurement-grouped, battery leave-one-battery-out). Environment as in
wave 1, plus one CPU lesson recorded in §15.

## 13. FD001 — GRU vs the shipped LSTM (Task B)

Identical shipped pipeline, split and recipe; only the recurrent cell
changes (GRU 128→64, same head). Local CPU, early stop at epoch 17
(best val epoch 7), train 583 s.

| Model | Val RMSE (all windows) | Test RMSE | Test MAE | Test NASA |
|---|---|---|---|---|
| LSTM (shipped, Kaggle GPU) | 13.01 | 13.73 | 9.89 | 431.3 |
| **GRU (wave 4, local CPU)** | **12.53** | **13.22** | **9.67** | **305.9** |

Reading: on this seed and protocol the GRU is marginally ahead of the
shipped LSTM on every metric, with ~25% fewer recurrent parameters.
That is a single-seed, single-split comparison — it shows the GRU is a
credible alternative cell for this data, not that GRUs are "better"
in general; the shipped LSTM remains the deployed model (its ONNX /
Rust serving path is built and parity-tested).

## 14. FD003 — CNN-LSTM vs LSTM (Task B), checkpoint saved

Conv1d frontend (15→64→32, k=3) feeding the shipped LSTM stack, on
FD003's own 15-feature pipeline. Early stop at epoch 21, train 392 s.

| Model | Val RMSE | Test RMSE | Test MAE | Test NASA |
|---|---|---|---|---|
| LSTM (wave 1) | **11.02** | 15.04 | 10.51 | 451.8 |
| **CNN-LSTM (wave 4)** | 11.54 | **13.17** | **9.10** | **290.5** |

Reading: the ranking flips between validation and the official test —
the plain LSTM's best-val checkpoint is better on val, the CNN-LSTM
generalises better to the test engines. Both numbers are reported;
picking the CNN-LSTM "because test" would be test-set selection, so
the honest claim is narrower: the convolutional frontend did not hurt
and its test-set degradation gap (val→test) is smaller here.
**Checkpoint saved** — `platform/artifacts/fd003_cnn_lstm.pt` plus a
preprocessing sidecar (features + scaler, same contract shape as
`ml/artifacts/preprocessing.json`) — which unblocks the wave-5
FD003 → FD001 transfer that wave 2 had to skip.

## 15. FD002 — tree regressors on flattened windows (Task B)

Random Forest and XGBoost regressors on flattened (30 × 23) windows,
official last-window protocol. Training windows subsampled to 10,000
of 37,051 (uniform, seed 42; val/test not subsampled) — recorded in
both records. The RF configuration itself had to be reduced for this
VM: three attempts (300 trees, max_features 1.0 at 20k/10k subsample;
300 trees, max_features 0.3 at 10k) each failed to finish the fit
stage in ~15 min of wall time on the throttled shared CPU, so the
recorded RF is 100 trees, max_features 0.3 (train 295 s; XGBoost's
histogram method needed no reduction, 75 s). The full story is in the
record's `cpu_feasibility_note`.

| Model | Val RMSE | Test RMSE | Test MAE | Test NASA |
|---|---|---|---|---|
| Ridge (wave 2) | 17.01 | 16.58 | 13.27 | 1267.2 |
| LSTM (wave 2) | 17.04 | **16.24** | **11.93** | **1099.8** |
| XGBoost reg (wave 4) | 19.29 | 17.49 | 13.69 | 1710.2 |
| Random Forest reg (wave 4) | 20.66 | 18.44 | 14.90 | 1979.2 |

Reading: flattened-window trees lose to both the linear baseline and
the LSTM on FD002. Flattening destroys the temporal order the trees
cannot recover from raw lags, and the 10k subsample further limits
them; the result is reported as measured, with the subsample caveat
attached — it should not be quoted as "trees can't do RUL", only as
"this flattened, subsampled configuration underperforms here".

## 16. AI4I — SVM and MLP complete the classifier set (Task A)

Same stratified split as wave 1 (test: 2,000 rows, 68 failures;
all-negative baseline accuracy 0.9660).

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Confusion [[TN,FP],[FN,TP]] |
|---|---|---|---|---|---|---|
| LightGBM (wave 1, best F1) | 0.986 | — | 0.765 | **0.782** | 0.970 | — |
| MLP (64,32) (wave 4) | 0.980 | 0.700 | 0.721 | 0.710 | **0.976** | [[1911, 21], [19, 49]] |
| SVM RBF, balanced (wave 4) | 0.917 | 0.279 | **0.912** | 0.428 | 0.961 | [[1772, 160], [6, 62]] |

Reading: the MLP has the best ranking quality (AUC 0.976) but a lower
F1 than LightGBM at the 0.5 threshold. The balanced SVM is a
recall-first detector: it catches 62/68 failures but raises 160 false
alarms (precision 0.28) — on a real line that trade might be chosen
deliberately, but as an F1 competitor it is far behind. The AI4I
synthetic-data caveat from §1 applies unchanged.

## 17. SECOM — One-Class SVM and autoencoder (Task C)

Both fitted on PASS rows of the train split only (1,170 rows), same
450 surviving features and split as wave 1; fail-rate prior
104/1567 ≈ 0.0664 sets the OCSVM ν and the autoencoder's threshold
quantile. Test: 314 rows, 21 fails.

| Model | Precision | Recall | F1 | False alarm rate | Confusion [[TN,FP],[FN,TP]] |
|---|---|---|---|---|---|
| Isolation Forest (wave 1) | — | 0.238 | **0.244** | 0.051 | — |
| One-Class SVM (wave 4) | 0.091 | 0.143 | 0.111 | 0.102 | [[263, 30], [18, 3]] |
| Autoencoder (wave 4) | 0.050 | 0.048 | 0.049 | 0.065 | [[274, 19], [20, 1]] |

Reading: neither newcomer beats the wave-1 Isolation Forest. The
autoencoder is the weakest: at the threshold implied by the fail-rate
prior it flags ~6.5% of passes yet catches only 1 of 21 fails —
reconstruction error on 450 anonymised, heavily-imputed channels is a
poor anomaly score here. With 21 test fails, all three F1 estimates
are high-variance; the defensible conclusion is that SECOM's
unsupervised signal is weak for every method tried so far, not that
one algorithm is proven superior.

## 18. CWRU — the recording-wise re-evaluation (Task A)

Wave 2's CWRU score (accuracy/F1 = 1.000) came from a window-wise
split over 4 recordings — every test window's recording was also in
training. Wave 4 downloads 8 more official recordings (0.014″ and
0.021″ faults at 0 hp; normal baselines at 1 and 2 hp — each file's
identity self-verified by its internal `X<id>_DE_time` variable;
99.mat as served bundles X098 alongside X099, and the self-identified
X099 series is the one used) for **12 recordings, 3 per class**, and
evaluates the same Random Forest (same features) by
**leave-one-recording-out**: pooled out-of-fold accuracy **0.665**,
F1-macro **0.421**.

Per held-out recording (accuracy): normal_0/1/2 = 1.000 / 1.000 /
1.000; B007 = 1.000, B014 = 0.373, B021 = 0.949; IR007/IR014/IR021 =
0.000 / 0.000 / 0.000; OR007/OR014 = 0.000 / 0.000, OR021 = 0.305.

Pooled confusion (rows = true): Normal [591, 0, 0, 0]; InnerRace
[0, 0, 122, 55]; Ball [3, 19, 137, 18]; OuterRace [1, 100, 58, 18].

Reading — the most informative result of this wave: **healthy vs
faulty transfers perfectly** across recordings (Normal recall 1.000),
and Ball partially (F1 0.555), but **fault location does not
transfer at all for InnerRace** (0/177 correct; IR windows are read
as Ball or OuterRace) and barely for OuterRace (F1 0.134). With only
time-domain amplitude statistics, *which* component is failing is
entangled with *how big* the fault is, so an unseen diameter of one
fault looks like a known diameter of another. The standard remedy is
frequency-domain evidence (envelope spectra at the characteristic
fault frequencies), which this feature set lacks — that, not more
trees, is the honest next step for CWRU. The wave-2 1.000 stands in
its record with its caveat; this record supersedes it as the number
to quote for CWRU diagnosis.

## 19. NASA Battery — leave-one-battery-out RUL (Task B)

Target: discharge cycles until capacity first falls below 1.4 Ah
(70% of the 2 Ah rating). LOBO over B0005/B0006/B0018; **B0007
excluded as censored** (never crosses 1.4 Ah — last recorded
1.432 Ah — so it has no observed EOL label). Random Forest on
cycle index + per-cycle discharge statistics; a small LSTM on
10-cycle capacity windows (first 10 cycles of each held-out battery
unscored — no full window).

| Held out | RF RMSE | RF MAE | LSTM RMSE | LSTM MAE |
|---|---|---|---|---|
| B0005 (168 cycles) | 11.87 | 9.62 | 13.45 | 9.38 |
| B0006 (168 cycles) | 7.13 | 4.86 | 19.82 | 11.56 |
| B0018 (132 cycles) | 7.77 | 6.01 | 7.49 | 5.45 |
| **Pooled** | **9.27** | **6.90** | 14.92 | 9.08 |

Reading: the Random Forest is the better battery model here, and it
is not close on B0006, where the capacity-only LSTM overfits the two
training batteries' fade shapes. Each LOBO fold trains on just two
batteries (~300 cycles), so both models are data-starved by
construction; the per-battery rows, not the pooled row, are the
honest unit of reporting. Errors of ~7–12 cycles on lives of
97–125 cycles are a reasonable first benchmark for this dataset in
the platform, not a validated prognostic.

## 20. Paderborn — measurement-grouped Random Forest (Task A)

30,022 windows from 240 measurements (3 bearings × 80), split by
**GroupShuffleSplit on measurement id** (192 train / 48 test
measurements) — no measurement on both sides. Accuracy **0.950**,
F1-macro **0.952** (train 28 s). Per-class F1: Normal 0.928,
InnerRace 0.937, OuterRace 0.992; the dominant confusion is
Normal ↔ InnerRace (108 + 164 windows).

Reading, with its binding caveat: there is still only **one physical
bearing per class**, so this measures within-bearing separability
across measurements and operating conditions — a stricter and more
honest number than a window-wise split would give, but not evidence
about unseen bearings. Paderborn data is CC BY-NC 4.0
(non-commercial use only).

## 21. Wave-4 files, and what remains for waves 5–6

```
platform/platform_lib/benchmarking/run_wave4.py    all wave-4 runners
platform/tests/test_wave4.py                       wave-4 tests (suite 64/64)
platform/artifacts/fd003_cnn_lstm.pt               FD003 CNN-LSTM checkpoint (wave-5 transfer)
platform/artifacts/fd003_cnn_lstm_preprocessing.json  its feature/scaler sidecar
platform/experiments/                              12 new records (29 total) + INDEX.json
platform/data/raw/cwru/{98,99,169,185,197,209,222,234}.mat  official CWRU downloads (gitignored)
```

Run commands (from `platform/`):
`../.venv/bin/python -m platform_lib.benchmarking.run_wave4 --set gru|cnnlstm|fd002trees|ai4i|secom|cwru|battery|paderborn`

Remaining for waves 5–6: FD003 → FD001 zero-shot transfer (now
unblocked by the saved checkpoint); FD002 → FD004 transfer (both
six-condition, FD004 adds the fan fault mode); SHAP/permutation for
the new best models; CWRU frequency-domain features (the §18
finding); per-regime normalisation for FD002/FD004 (wave-2 finding,
still open); dashboard integration of the platform results.

---

# Waves 5–6 (2026-11-15) — cross-dataset generalisation, IMS, XAI

Wave 5 completes the transfer matrix in both directions (the
wave-2 FD001 → FD003 result stood alone), integrates IMS, and wave 6
adds the explainability records for the wave-4 models. All runs are
seed 42, source-fitted preprocessing only, and every record states
its protocol. `FAULT_DIAGNOSIS.md` (new this wave) consolidates what
the platform can and cannot claim, record by record.

## 22. The transfer matrix — in-domain vs cross-domain, side by side

| Pair | Direction | Model | In-domain (target) | Cross-domain | Δ RMSE / acc |
|---|---|---|---|---|---|
| C-MAPSS RUL | FD001 → FD003 | shipped LSTM (14 feat) | 15.04 | **48.80** | +224% |
| C-MAPSS RUL | FD003 → FD001 | CNN-LSTM (15 feat) | 13.73 | **15.82** | +15% |
| C-MAPSS RUL | FD002 → FD004 | Ridge (23 feat) | 19.43 | **51.38** | +164% |
| C-MAPSS RUL | FD004 → FD002 | Ridge (23 feat) | 16.58 | **18.07** | +9% |
| Bearings | CWRU → Paderborn | RF (10 feat) | 0.950 acc | **0.253 acc** | −0.70 |
| Bearings | Paderborn → CWRU | RF (10 feat) | 0.665 acc | **0.125 acc** | −0.54 |

Records: `cmapss_fd003_to_fd001__taskB__transfer_cnn_lstm__s42`,
`cmapss_fd002_to_fd004__taskB__transfer_ridge__s42`,
`cmapss_fd004_to_fd002__taskB__transfer_ridge__s42`,
`cwru_to_paderborn__taskA__transfer_rf__s42`,
`paderborn_to_cwru__taskA__transfer_rf__s42` (plus the wave-2
FD001 → FD003 record).

## 23. FD003 → FD001 — the asymmetry, measured

The FD003 CNN-LSTM, applied zero-shot to FD001 with FD003's 15
features and FD003's scaler, scores RMSE **15.816** / MAE 11.420 /
NASA **395.0** — against 48.80 for the reverse direction. It is
well calibrated on average (mean prediction 73.5 vs mean truth
74.4; 13 of 100 engines off by more than 25 cycles), and its NASA
score is even below the shipped model's in-domain 431.3 — that is
a property of the asymmetric NASA penalty (the transfer model's
errors are comparatively symmetric), **not** evidence it beats the
in-domain model: its RMSE/MAE are worse than both in-domain FD001
references (13.73 shipped LSTM, 13.22 GRU).

Why the asymmetry is believable, stated with its confound: FD003
training contains both fault modes (HPC + fan) and FD001 only one
(HPC), so the FD003 model has seen a superset of the target's
failure physics, while the FD001 model prices FD003's fan mode as
health. The confound: the two directions also differ in
architecture (LSTM vs CNN-LSTM) and feature set (14 vs 15, sensor_6
kept for FD003). A perfectly controlled pair would retrain both
directions with one architecture; the direction of the effect is
consistent with §24's Ridge pair, which *is* architecture- and
feature-controlled, so the fault-mode explanation is the one the
evidence supports.

## 24. FD002 ↔ FD004 Ridge — the controlled pair

Both subsets derive the identical 23-feature set (verified in-run),
Ridge is deterministic (the refit *is* the wave-2 model), so this
pair isolates distribution/fault-mode shift from everything else.

- FD002 → FD004: RMSE **51.380** / MAE 36.071 / NASA 7,095,957.
  Failure anatomy: 62 of 237 engines over-predicted by > 50 cycles;
  the worst (engine 235: true 24 → predicted 160; engine 135: true
  7 → 151) show the FD002 model reading FD004's fan-mode engines as
  early-life. NASA explodes because the score punishes late
  predictions exponentially — a handful of +130-cycle errors
  dominates it.
- FD004 → FD002: RMSE **18.069** / MAE 15.182 / NASA 1,266 — a mild
  +9% over FD002's in-domain Ridge (16.58).

Same conclusion as §23, without the confound: **wide → narrow
transfers, narrow → wide fails, and the failure mode is dangerous**
(over-predicting the life of engines failing in an unseen mode).

## 25. CWRU ↔ Paderborn — fault location does not transfer

Both directions fail outright. CWRU → Paderborn accuracy **0.253**
(F1-macro 0.205) is below the target's 0.334 majority-class rate;
Paderborn → CWRU accuracy **0.125** (F1-macro 0.156) against a
0.625 majority rate. The confusion matrices show why: **Normal is
never predicted correctly in either direction (F1 = 0.0)** — a
CWRU-trained forest calls every Paderborn window Inner or Outer
race; a Paderborn-trained forest calls all 591 CWRU Normal windows
OuterRace. Identical features and label names cannot compensate
for 12 kHz vs 64 kHz sampling (2048-sample windows = 170.7 ms vs
32.0 ms), different rigs, bearings, speeds and defect creation.
On time-domain features, "Normal" is a property of the rig. This
closes the bearing-transfer question for this feature set;
frequency-domain features are the documented next attempt, not a
promised fix.

## 26. IMS Bearing — integrated; alarm timing on Test 1

The user approved lifting wave 3's size gate; the official zip
(1,075,597,174 B, SHA-256 `21001ac2…ad83`) is triple-nested
(zip → IMS.7z → per-test .rar — the wave-3 community note was
right). Test 1 extracted: 2,156 snapshots × (20,480 × 8) at 20 kHz.
New `IMSAdapter` emits per-snapshot per-channel features.

Per-bearing Isolation Forests (fit on the first 30% = 646
snapshots; threshold = train q95; alarm = 10 consecutive snapshots
above) — record `ims__taskC__isolation_forest__s42`:

| Bearing | Teardown truth | First sustained alarm | Last-10% exceedance |
|---|---|---|---|
| 1 | none reported | never | 0.10 |
| 2 | none reported | never | 0.19 |
| 3 | inner-race defect | **1819 / 2156** (2003-11-22) | 1.00 |
| 4 | roller-element defect | **1433 / 2156** (2003-11-18) | 1.00 |

The detectors fire on exactly the two bearings that failed —
bearing 4 gives 723 snapshots (≈ 5.0 days) of warning, bearing 3
gives 337 (≈ 2.3 days) — and never sustain an alarm on the two
healthy bearings (their late-run exceedance rates of 0.10/0.19 are
isolated excursions, never 10 in a row). No precision/recall is
reported: IMS has no per-snapshot labels, and inventing them would
be fabrication. DATASET_REGISTRY.md now lists IMS as integrated
(Test 1); Tests 2–3 remain unextracted in the local archive.

## 27. Wave-6 XAI — Battery SHAP, Paderborn permutation

**Battery SHAP** (`battery__taskB__shap_random_forest__s42`):
explanation model = RF refit on all cycles of B0005/B0006/B0018
(protocol difference vs LOBO stated in the record; its in-sample
RMSE 1.09 is labelled in-sample and is not a performance claim).
Mean |SHAP| on B0005: **soh 15.99, capacity_ah 15.05** — two views
of the same capacity trajectory — then duration_s 7.29, and every
voltage/temperature feature ≤ 1.37 (ambient_temperature exactly
0.0). The model's RUL is a capacity-curve extrapolation; SHAP
makes that plain rather than dressing it up.

**Paderborn permutation** (`paderborn__taskA__permutation_rf__s42`):
the refit reproduces wave 4's grouped-split accuracy (0.9499 vs
0.950). Importance (accuracy drop): **kurtosis 0.196, skewness
0.164**, then a cliff to std 0.038 / RMS 0.038 / crest 0.023;
`max` is ≈ 0 (−0.002). The forest separates the three classes
almost entirely on impulsiveness statistics — physically
consistent with localised race defects, and a useful pointer for
the frequency-domain follow-up (envelope spectra are the natural
extension of the same idea).

## 28. Wave-5/6 files, and what remains

```
platform/platform_lib/adapters/ims.py                 IMS adapter (Test 1)
platform/platform_lib/benchmarking/run_wave5.py       transfers + XAI runners
platform/platform_lib/benchmarking/run_wave5_ims.py   IMS runner
platform/FAULT_DIAGNOSIS.md                           capability/limits audit
platform/tests/test_wave5.py                          wave-5/6 tests (suite 96/96)
platform/experiments/                                 8 new records (37 total) + INDEX.json
platform/data/raw/ims/                                official archive + Test 1 (gitignored)
```

Run commands (from `platform/`):
`../.venv/bin/python -m platform_lib.benchmarking.run_wave5 --set fd003tofd001|ridge|bearings|battery_shap|paderborn_perm|ims`

Remaining: IMS Tests 2–3 (in the local archive, unextracted);
per-regime normalisation for FD002/FD004 (open since wave 2);
CWRU/Paderborn frequency-domain features for a second transfer
attempt; Paderborn's real-damage bearings (artificial → real shift,
not yet downloaded); a controlled LSTM↔CNN-LSTM transfer pair to
remove §23's architecture confound; dashboard integration of the
platform results.

---

# Advanced expansion — Waves 1–8, closing note (2026-11-15)

The advanced expansion brief (Waves 1–8) is complete. Waves 1–3 built
the dataset registry and integrated NASA Battery + Paderborn (IMS and
MFPT blocked, for the reasons recorded in the registry); Wave 4
(§§13–21) expanded the model set and replaced CWRU's leaky headline
with the recording-wise evaluation; Waves 5–6 (§§22–28) ran the
cross-dataset transfer matrix, integrated IMS Test 1, and added the
Wave-6 XAI records plus `FAULT_DIAGNOSIS.md`; Wave 7 fed these records
into four new dashboard views (Platform, Datasets, Experiments,
Research) alongside the original FD001 views; Wave 8 re-verified
everything and completed the documentation.

Wave-8 verification (all run locally, 2026-11-15): platform suite
**96/96 passed**; original Python suite (the CI command over
`ml/tests`, `maintenance`, `simulator`) **23/23 passed**; Rust edge
suite **15/15 passed**, including the Rust↔Python ONNX parity test;
dashboard production build green. `DATASET_RESEARCH.md` — found
truncated in Wave 3 — has been fully rewritten from the registry, and
the root README gained a platform section.

Still open (unchanged from §28, restated so nothing is silently
dropped): IMS Tests 2–3 (in the local archive, unextracted);
per-regime normalisation for FD002/FD004; frequency-domain bearing
features for a second CWRU↔Paderborn transfer attempt; Paderborn's
real-damage bearings; a controlled LSTM↔CNN-LSTM transfer pair to
remove §23's architecture confound.

---

# Wave 9 (2026-11-15) — frequency-domain bearing features + IMS completion

Wave 9 attacks the two open problems named in the Waves 1–8 closing
note: the missing frequency-domain bearing features (re-running the
two honest protocols of waves 4–5 with features as the ONLY changed
variable) and IMS Tests 2–3. The headline, stated before the tables:
**the spectral features did not fix either bearing problem, and the
identical IMS protocol that succeeded on Tests 1–2 misses Test 3's
failed bearing.** All three results are reported as measured; no
protocol was re-tuned after seeing an outcome.

## 29. The wave-9 feature set — design and the rate-freedom choice

New module `platform_lib/features/spectral.py`. The combined set is
the untouched 10 time-domain statistics plus 14 spectral/envelope
features (24 total): dominant frequency, spectral centroid, spectral
entropy, energy in 8 log-spaced bands, and Hilbert-envelope features
(envelope-spectrum peak amplitude and frequency, envelope kurtosis).

Binding design choice: CWRU records at 12 kHz and Paderborn at
64 kHz, so every spectral quantity is expressed in RATE-FREE units —
frequencies and band edges as fractions of Nyquist (bands over
0.005–1.0 × Nyquist), band energies normalised by total spectral
energy, envelope peak amplitude normalised by total envelope
magnitude. The identical definitions therefore apply to both
datasets. The records also state the limit of this choice: it
removes the sampling-rate confound only. Shaft speeds differ
independently of sampling rates, so defect frequencies expressed
relative to shaft rotation are NOT aligned, and a 2048-sample window
still spans 170.7 ms of rotation on CWRU vs 32.0 ms on Paderborn.

## 30. CWRU recording-wise, re-run with spectral features (Task A)

Record: `cwru__taskA__random_forest_spectral_recordingwise__s42`.
Same 12 recordings, same leave-one-recording-out protocol, same RF
(300 trees, balanced), same seed as §18 — features are the only
change.

| Metric | Wave 4 (time only) | Wave 9 (combined) | Δ |
|---|---|---|---|
| Accuracy | 0.665 | 0.679 | +0.014 |
| F1-macro | 0.421 | 0.417 | −0.004 |
| InnerRace correct | 0 / 177 | 15 / 177 | — |
| InnerRace F1 | 0.000 | 0.132 | +0.132 |
| OuterRace F1 | 0.134 | 0.008 | −0.126 |
| Ball F1 | 0.555 | 0.534 | −0.021 |
| Normal F1 | 0.997 | 0.996 | −0.001 |

Honest reading: this is a reshuffle, not a fix. Accuracy moved
+0.014 while F1-macro went slightly DOWN. InnerRace is no longer
literally never recognised (15/177, recall 0.085), but OuterRace
collapsed in exchange (1/177 correct). Per-recording detail (in the
record) shows the same structure as wave 4: Normal and Ball
recordings largely recognised, fault-location classes confused with
each other on unseen recordings. With three recordings per class,
each held-out fold is also a new fault diameter — the spectral
signature of a 0.007″ defect evidently does not pin down the class
of a 0.021″ one under these features either. **Conclusion: fault
location still does not generalise across recordings on CWRU with
window-level spectral features, just as it did not with time-domain
features.**

## 31. CWRU ↔ Paderborn transfer, re-run with spectral features (Task A)

Records: `cwru_to_paderborn__taskA__transfer_rf_spectral__s42`,
`paderborn_to_cwru__taskA__transfer_rf_spectral__s42`. Same zero-shot
protocol as §25 (all-source → all-target, shared 3-class label set).

| Direction | Metric | Wave 5 (time only) | Wave 9 (combined) | Δ |
|---|---|---|---|---|
| CWRU → Paderborn | accuracy | 0.253 | 0.210 | −0.043 |
| | F1-macro | 0.205 | 0.171 | −0.034 |
| | Normal F1 | 0.000 | 0.195 | +0.195 |
| Paderborn → CWRU | accuracy | 0.125 | 0.187 | +0.062 |
| | F1-macro | 0.156 | 0.286 | +0.130 |
| | Normal F1 | 0.000 | 0.000 | 0.000 |

Honest reading, per direction:

- **CWRU → Paderborn got worse overall**, with one genuine change:
  Normal is partially visible for the first time (F1 0.195, recall
  0.255 — a quarter of Paderborn's healthy windows recognised),
  while InnerRace collapsed to 13 / 10,001 correct. The wave-5
  "Normal is a property of the rig" finding is dented in this
  direction, not overturned.
- **Paderborn → CWRU improved on paper, but the improvement is one
  class deep**: InnerRace is recognised perfectly (177/177,
  F1 0.857) while Normal (0/591) and OuterRace (0/177) are missed
  entirely. A classifier that finds exactly one of three classes
  has not transferred; it has found one coincidentally aligned
  signature.

**Conclusion: cross-rig bearing transfer remains a failure in both
directions with spectral features.** Removing the sampling-rate
confound was necessary-but-not-sufficient; the remaining confounds
(shaft-speed-relative frequencies, rig/bearing/sensor differences,
5.3× window time span) dominate. The principled next attack — not
attempted here — is shaft-speed-normalised features (order tracking
/ characteristic-frequency ratios), which requires the speed channel
Paderborn carries and CWRU's per-recording RPM metadata.

## 32. IMS Tests 2 and 3 — the protocol completes the dataset, and misses once (Task C)

Records: `ims_test2__taskC__isolation_forest__s42`,
`ims_test3__taskC__isolation_forest__s42`. Protocol IDENTICAL to
§26 (constants and alarm helper imported from the wave-5 runner):
per-bearing Isolation Forest on the first 30% of snapshots,
train-q95 threshold, alarm = first run of ≥10 consecutive
exceedances. One documented data difference: Tests 2–3 recorded
ONE accelerometer per bearing (4 channels), so each detector uses
5 features instead of Test 1's 10. Ground truth was verified
against the Readme Document inside the official archive (zip
SHA-256 re-verified, exact match): Test 2 — bearing 1 outer-race
failure; Test 3 — bearing 3 outer-race failure.

**Test 2** (984 snapshots, 2004-02-12 → 2004-02-19):

| Bearing | Documented defect | First sustained alarm | Last-decile exceedance |
|---|---|---|---|
| 1 | **outer-race failure** | **532** (452 before end) | 1.000 |
| 4 | none reported | 655 | 1.000 |
| 2 | none reported | 722 | 1.000 |
| 3 | none reported | 879 | 1.000 |

The failed bearing alarms FIRST, 452 snapshots (≈3.1 days) before
the run ends — the success ordering. The honest caveat: unlike
Test 1, the healthy bearings also sustain alarms later, and by the
last decile every bearing exceeds its threshold 100% of the time —
on a shared shaft the failing bearing's vibration is not private,
and this protocol cannot attribute the end-of-run anomaly to one
bearing by exceedance alone, only by alarm ORDER.

**Test 3** (4,448 snapshots, 2004-03-04 → 2004-04-04):

| Bearing | Documented defect | First sustained alarm | Last-decile exceedance |
|---|---|---|---|
| 3 | **outer-race failure** | **never** | 0.196 |
| 4 | none reported | 2036 | 0.697 |
| 1 | none reported | 2325 | 0.034 |
| 2 | none reported | never | 0.306 |

**The protocol misses Test 3's failed bearing.** Bearing 3 never
sustains 10 consecutive above-threshold snapshots; its last-decile
exceedance (0.196) is above its 0.05 train baseline, so a weak
late-run elevation exists in the record, but it does not meet the
alarm bar that Tests 1–2 met — and the strongest late-run signal
sits on bearing 4, which has no reported defect (shaft coupling is
the standing caveat; no cause is claimed). The protocol was NOT
re-tuned after this result: loosening the sustained-run length or
threshold until bearing 3 fires would be fitting the evaluation to
the answer. Stated plainly: with per-snapshot time-domain channel
statistics and this detector, IMS stands at two tests where the
failed bearing is caught early (Tests 1–2) and one where it is not
(Test 3).

Archive quirk found during extraction: `3rd_test.rar` contains
6,324 files. The first 4,448 are exactly the Readme-documented
experiment (ending 2004-04-04 19:01:57 to the second); the
remaining 1,876 (to 2004-04-18) postdate the documented experiment
and teardown and are stored separately
(`3rd_test_beyond_documented_period/`), unused. The Readme does not
say what they are; nothing is assumed about them.

## 33. Wave-9 files, verification, and what remains

Files: `platform_lib/features/spectral.py` (new),
`platform_lib/benchmarking/run_wave9.py`,
`platform_lib/benchmarking/run_wave9_ims.py`,
`platform_lib/adapters/ims.py` (generalised to Tests 1–3; Test 1
behaviour unchanged), 5 new experiment records (42 total),
`tests/test_wave9.py`, `DATASET_REGISTRY.md` (IMS status: Tests
1–3). Verification: full platform suite **116/116 passed**
(96 at wave 8 + 20 wave-9 additions), including the global schema
test over all 42 records and the IMS alarm-timing framing tests.

Still open (unchanged items restated): per-regime normalisation for
FD002/FD004; Paderborn's real-damage bearings; the controlled
LSTM↔CNN-LSTM transfer pair; shaft-speed-normalised (order-tracked)
bearing features as the next transfer attempt; an IMS detector
variant study (e.g. spectral snapshot features or a different
threshold rule) — to be run as a NEW protocol with its own record,
never as a silent re-tune of the wave-5/9 protocol.

# Wave 10 (2026-11-15) — per-regime normalisation + Paderborn real damage

## 34. Per-regime normalisation — method (Task B)

Wave 2 scored FD002/FD004 with ONE global z-score scaler, mixing six
operating regimes into a single mean/std per sensor. Wave 10 changes
ONLY the scaling (`platform_lib/regime_pipeline.py`): regimes are
derived from the TRAIN split's 3 setting columns by KMeans (k = 6,
seed 42, settings standardised by their train-split std first so the
small-range setting_2 is not swamped). The alternative — unique
rounded setting combinations — was measured first and rejected:
rounding to 2 decimals yields 11 combinations (jitter splits the
x.00/x.01 variants) and rounding to 1 yields 7; neither is the
documented 6-regime structure. The KMeans centroids land exactly on
the six documented nominal conditions for both subsets —
(0, 0, 100), (10, 0.25, 100), (20, 0.70, 100), (25, 0.62, 60),
(35, 0.84, 100), (42, 0.84, 100) — with train-row sizes
FD002 [6417, 6568, 6416, 10777, 6451, 6454] and
FD004 [12195, 7380, 7276, 7250, 7331, 7299] (the 42-condition holds
roughly double rows in both). One z-score scaler per regime is fit on
that regime's TRAIN rows and applied to the 20 sensor features;
val/test rows are assigned to the nearest train centroid. The 3
setting features keep the wave-2 global scaler — within a regime a
setting is near-constant by construction, so a per-regime z-score
for it is undefined. Split, feature derivation (23 features), cap
125, window 30, models and hyper-parameters are identical to wave 2;
each record carries the wave-2 numbers as its comparison block.

## 35. Per-regime results — before/after, as measured

Test set, last window per engine (wave-2 global → wave-10 per-regime):

| Subset | Model | RMSE | MAE | NASA | ΔRMSE | ΔNASA |
|---|---|---|---|---|---|---|
| FD002 | Ridge | 16.582 → **16.518** | 13.268 → **13.229** | 1267.2 → 1378.7 | −0.064 (−0.4%) | **+111.5 (worse)** |
| FD004 | Ridge | 19.425 → **19.115** | 15.708 → **15.223** | 1779.1 → **1766.0** | −0.310 (−1.6%) | −13.1 |
| FD002 | LSTM | 16.236 → **14.046** | 11.933 → **10.087** | 1099.8 → **1039.9** | **−2.191 (−13.5%)** | −59.9 |

The FD004 LSTM was NOT re-run (CPU budget, stated in advance); its
absence from the record set is deliberate, not an omission.

Honest reading: for the flattened Ridge the change is negligible on
RMSE/MAE and mixed on NASA — FD002's NASA score got WORSE (+111.5)
despite the slightly better RMSE, i.e. per-regime scaling shifted
the error distribution toward the late predictions NASA punishes.
For the FD002 LSTM the same change is a large, real gain on all
three metrics (−13.5% RMSE; best val RMSE 15.19, early stop at
epoch 12, 464 s train) — 14.046 is now the best FD002 test RMSE in
the platform (wave-2 LSTM 16.24, Ridge 16.58, XGBoost 17.49,
RF 18.44). Scope of the claim, stated plainly: one seed (42), the
LSTM measured on FD002 only. Whether it holds on FD004 or across
seeds is unmeasured, not assumed.

## 36. Paderborn — REAL-damage bearings (Task A)

Waves 3–4 used the artificial EDM-seeded bearings. Wave 10 adds the
dataset's REAL fatigue damage. Picks were verified from the OFFICIAL
per-bearing damage-profile PDFs shipped inside each official archive
(Lessmeier, KAt Paderborn) — read verbatim, not inferred from codes:

- **KA04** (OuterRace): 1 damage — mode "fatigue", symptom
  "Pitting", component OR, method "lifetime test", extent 1,
  2 × 3 mm, cause "operating conditions and lubricant (overload,
  wrong viscosity, contamination)".
- **KI04** (InnerRace): primary damage — mode "fatigue (below the
  surface)", symptom "Pitting", component IR, method "lifetime
  test", extent 1, 2 × 1 mm; PLUS a documented secondary damage —
  plastic deformation by foreign objects (particle-caused OR
  indentations). It is in the record, not hidden; the class follows
  the dataset's own primary-damage categorisation. (KI14's profile
  was also read: same primary+secondary pattern; archive downloaded,
  not extracted.)

RF, same 10 time-domain features, same measurement-grouped split as
wave 4 (192/48 measurements), K001 as Normal:

| | Accuracy | F1-macro |
|---|---|---|
| Wave 10 — REAL damage (KA04/KI04) | **0.9977** | **0.9978** |
| Wave 4 — artificial EDM (KA01/KI01), context only | 0.950 | 0.952 |

Confusion (rows = true): Normal 1869/1875 (6 → InnerRace),
InnerRace 2367/2375 (8 → Normal), OuterRace 1750/1750 (perfect).

This is NOT a controlled artificial-vs-real comparison — different
physical bearings, possibly different damage extents — so the
correct claim is narrow: under the identical protocol, the real
fatigue bearings in this scope are at least as separable as the
artificial ones were, with OuterRace perfectly separated. The
binding caveat is unchanged: one physical bearing per class, so
this is within-bearing separability across measurements, not
generalisation to unseen bearings. The actual artificial→real
transfer experiment (train on EDM bearings, test on fatigue
bearings) remains undone and is the natural next step.

## 37. Wave-10 files, verification, and what remains

Files: `platform_lib/regime_pipeline.py` (new),
`platform_lib/benchmarking/run_wave10.py`,
`platform_lib/adapters/paderborn.py` (KA04/KI04 added to BEARINGS;
default load unchanged — the original 3 bearings are now the
explicit DEFAULT_BEARINGS), 4 new experiment records (46 total),
`tests/test_wave10.py`, `DATASET_REGISTRY.md` (Paderborn status).
Verification: full platform suite **130/130 passed** (116 at wave 9
+ 10 wave-10 tests + 4 schema tests over the new records),
including train-only regime-derivation unit tests and the
damage-documentation record tests.

Still open: the FD004 per-regime LSTM; the artificial→real
Paderborn transfer; the controlled LSTM↔CNN-LSTM transfer pair;
order-tracked bearing features; an IMS detector variant study (as a
NEW protocol, never a silent re-tune).

# Waves 9–11, closing note (2026-11-15)

Waves 9–10 are documented in §§29–37 above. Wave 11 closed the loop
on presentation and verification — no new experiments. The dashboard
export (`dashboard/platform_export.py`) was refreshed and extended so
`dashboard/src/data/platform_data.json` now carries all **46
records** (it was generated at 37): 8 transfer records (the two
Wave-9 spectral transfers gained verdicts written from their
records), 14 best-result rows (adding the FD002 per-regime LSTM
14.046 as FD002's best Task-B row, the Paderborn real-damage row,
and IMS Test 2/3 alarm-timing rows whose display text is built from
the records — Test 3's row states the miss as a miss), and
Datasets-view record counts that reconcile exactly (38 card records
+ 8 transfers = 46). The Research view's two stale hardcoded
passages were corrected: the transfer pattern now counts all 8
records and notes the spectral outcome, and the IMS footer no
longer claims Tests 2–3 are unextracted — it states the Test-2
shared-shaft caveat and the Test-3 miss. README platform numbers
were synced to the same counted values. The CWRU window-split
exclusion from best results was re-verified after the refresh: the
CWRU best row remains the recording-wise record (F1-macro 0.421).

Final verification sweep (Wave 11, all run for real): platform
suite **130/130 passed**; original Python suite **23/23** (CI
command); Rust edge tests **15/15** (11 unit + 4 integration incl.
ONNX parity); dashboard production build **green** (only the
pre-existing >500 kB chunk advisory). Consistency audit: INDEX.json
entries == record files on disk == platform_data.json counts ==
README numbers (46 records; Task A 20 · B 20 · C 6; 10 of 20
registry datasets integrated).

# Wave 12 (2026-11-15) — FD004 per-regime LSTM + order-domain bearing features

Wave 12 executes the two principled items the earlier waves
documented but had not run: the FD004 per-regime LSTM (skipped in
wave 10 for CPU budget) and the order-domain (shaft-speed-relative)
bearing features — the documented next attack on the cross-rig
transfer failure of waves 5 and 9. Four new records, 50 total.
Everything below is as measured; the transfer verdict is stated
plainly at §40.

## 38. FD004 per-regime LSTM — the wave-10 protocol, completed (Task B)

Identical protocol to the wave-10 FD002 per-regime LSTM (same
regime pipeline — KMeans k=6 on train settings, per-regime sensor
z-scores, settings keep the global scaler — same LSTM
architecture/hyperparameters, cap 125, window 30, engine-wise
split seed 42); only the subset changed. Early stop at epoch 14
(best val RMSE 15.651 at epoch 4; train 841 s).

| FD004 run | Test RMSE | Test MAE | NASA |
|---|---|---|---|
| Wave-2 LSTM, global norm | 18.809 | 13.539 | 2141.5 |
| Wave-10 Ridge, per-regime | 19.115 | 15.223 | 1766.0 |
| **Wave-12 LSTM, per-regime** | **15.372** | **10.864** | **1457.4** |

Per-regime normalisation helps the FD004 LSTM even more than the
FD002 one: −3.44 RMSE (−18.3%) and −684 NASA vs its own global-norm
run, and −3.74 RMSE (−19.6%) vs the per-regime Ridge. Two
consequences worth stating: (i) the per-regime LSTM gain is now
confirmed on BOTH 6-condition subsets (FD002 −13.5%, FD004
−18.3%), not a single-subset accident; (ii) on FD004 the LSTM
retakes the NASA lead it had lost to Ridge in wave 2 (1457 <
1766) — the wave-2 reading "on FD004 the LSTM loses NASA to
Ridge" was a property of the global normalisation, not of the
model class. The claim remains scoped to seed 42.

## 39. Order-domain features — the physics (Task A)

Bearing faults repeat at characteristic multiples of shaft speed.
Wave 9's spectral features were rate-free (fractions of Nyquist)
and by their own design note did not align these orders. Wave 12
reads the Hilbert-envelope spectrum at the characteristic orders
of the ACTUAL bearing in each rig, per recording speed:

| Bearing | Geometry (source) | BPFO | BPFI | BSF* | FTF |
|---|---|---|---|---|---|
| CWRU SKF 6205-2RS JEM | 9 balls, Bd 0.3126 in, Pd 1.537 in (CWRU Data Center) | 3.5848 | 5.4152 | 4.7134 | 0.3983 |
| Paderborn 6203 | 8 balls, Bd 6.75 mm, Pd 28.55 mm (nominal geometry) | 3.0543 | 4.9457 | 3.9932 | 0.3818 |

\*BSF here is the ball-DEFECT order 2×ball-spin (a ball defect
strikes both raceways once per revolution; it is also the value
CWRU publishes as its rolling-element multiplier, 4.7135). Orders
are computed from the kinematic formulas in
`platform_lib/features/order.py`; for the 6205 they reproduce
CWRU's own published multipliers to ≤8e-5, for the 6203 they
match the nominal-geometry values cited in the Paderborn
literature exactly at 4 dp. Speeds: CWRU per recording from the
in-file X\<id\>RPM scalar where shipped (10 of 12 recordings,
1796–1797 rpm), else the CWRU catalogue nominal for the load
(98.mat → 1772, 99.mat → 1750); Paderborn per measurement from
the condition code (N15 → 1500, N09 → 900), cross-checked against
the recorded speed channel (means 1499.6 / 899.7 rpm).

Features (9, added to the untouched 10 time-domain): envelope-
spectrum energy share in a ±2% band at each of BPFO/BPFI/BSF/FTF
and its 2nd harmonic (nearest single bin if the band holds no
bin centre), plus order-domain spectral entropy over orders
(0, 12]. The ±2% band exists because slip moves true defect
rates ~1–2% off the kinematic orders.

Resolution reality, stated before the results: true order
resolution is 1/(revolutions per 2048-sample window) — 0.196
orders on CWRU (5.11 rev/window) but **1.25 orders at Paderborn
N15 and 2.08 at N09** (0.80 / 0.48 revolutions per window),
coarser than the spacing between the characteristic orders
themselves. On the Paderborn side the order features are blunt
by construction of the shared window protocol.

## 40. Order-domain results — a real in-domain win, and the transfer question answered (Task A)

CWRU recording-wise (LOGO over the same 12 recordings; RF
identical; only the feature set changes):

| Feature set | Accuracy | F1-macro | InnerRace correct |
|---|---|---|---|
| Wave 4: time-domain (10) | 0.665 | 0.421 | 0/177 |
| Wave 9: + spectral (24) | 0.679 | 0.417 | 15/177 |
| **Wave 12: + order (19)** | **0.802** | **0.689** | **109/177** |

The order features are the first change that materially moves
fault LOCATION in-domain: InnerRace F1 0.760 (109/177, precision
0.991), Ball 141/177, Normal 591/591. The remaining weakness is
OuterRace (59/177 — all of them from the 0.007″ recording; the
0.014″ and 0.021″ OR recordings score 0.000 per-recording, and
117 OR windows are read as Ball). Per-recording detail is in the
record.

Cross-rig zero-shot transfer (same protocols as waves 5/9):

| Direction | Wave 5 (time) | Wave 9 (spectral) | Wave 12 (order) |
|---|---|---|---|
| CWRU → Paderborn acc / F1-macro / Normal F1 | 0.253 / 0.205 / 0.000 | 0.210 / 0.171 / 0.195 | 0.335 / 0.169 / **0.000** |
| Paderborn → CWRU acc / F1-macro / Normal F1 | 0.125 / 0.156 / 0.000 | 0.187 / 0.286 / 0.000 | 0.163 / 0.206 / **0.000** |

Read as measured, not as the accuracy row alone suggests: the
CWRU → Paderborn "improvement" to 0.335 is degenerate — the
forest predicts OuterRace for 29,955 of 30,022 windows, i.e.
accuracy equals the majority-class rate (0.334) while F1-macro
is the worst of the three attempts. In the other direction all
591 CWRU Normal windows are again predicted OuterRace (the
InnerRace recall of 0.537 is real signal, but a rig's Normal
class is never recognised). **Normal F1 is 0.000 in both
directions under all three feature families.**

**Closing statement on the bearing-transfer question.** Order
alignment works exactly where the physics says it should — on a
fixed rig, where it produced the platform's best honest CWRU
result (0.802). Across rigs, three successive feature families
(time-domain, rate-free spectral, order-domain) have now failed
the same way, and the order features were the principled best
case for hand-crafted alignment. For window-statistic features
with a Random Forest, the cross-rig question is therefore
answered, negatively: the blocker is not frequency alignment.
It is the measurement regime itself — sub-revolution windows on
the Paderborn side (order resolution coarser than the order
spacing), different sensors, mountings and rig transfer
functions, and different fault geometries — none of which a
per-window feature can remove. What remains genuinely open is
a different method class (learned representations on raw
waveforms, longer windows, explicit domain adaptation), not a
fourth hand-crafted feature set.

## 41. Wave-12 files, verification, and what remains

New: `platform_lib/features/order.py` (BearingSpec + order
features, geometry and speed provenance in the docstring),
`platform_lib/benchmarking/run_wave12.py`, records
`cmapss_fd004__taskB__lstm_perregime__s42`,
`cwru__taskA__random_forest_order_recordingwise__s42`,
`cwru_to_paderborn__taskA__transfer_rf_order__s42`,
`paderborn_to_cwru__taskA__transfer_rf_order__s42`,
`tests/test_wave12.py` (geometry-vs-published checks; synthetic
AM test — a signal modulated at the BPFO order puts >10× more
band energy at BPFO than BPFI, and vice versa; comparison-block
and baseline-carrying checks on all four records).

Verification: full platform suite **142/142 passed** (130 at
wave 11 + 12 wave-12 tests, including the schema tests over the
four new records). Frame counts asserted in the runner against
the wave-9 records before any fit (CWRU 1,122 windows; transfer
frames 945 / 30,022) — no drift.

Still open: the artificial→real Paderborn transfer; the
controlled LSTM↔CNN-LSTM transfer pair; an IMS detector variant
study (as a NEW protocol, never a silent re-tune); cross-rig
bearing transfer via learned representations (a new method
class — see §40 for why hand-crafted features are closed).
