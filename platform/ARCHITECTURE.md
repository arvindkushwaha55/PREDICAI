# PREDICAI Platform — Architecture (Phase 2)

**Status:** Design + first-wave implementation (Phase 4 benchmarks run under this design).
**Companion docs:** `DATASET_RESEARCH.md` (Phase 1), `EXPERIMENTS_REPORT.md` (measured results).

The platform generalises the working FD001 system (PyTorch LSTM → ONNX → Rust/Axum
edge API → React dashboard) into a multi-dataset, multi-model predictive-maintenance
platform **without modifying that system**. Everything new lives under `platform/`;
the FD001 pipeline becomes one registered entry (FD001 × Task B) inside the platform
rather than being rewritten.

---

## 1. The three tasks and their datasets

| Task | Definition | Output | Datasets (wave 1 in **bold**) |
|------|-----------|--------|-------------------------------|
| **A — Failure classification** | Supervised binary/multi-class prediction of a failure or fault label from a snapshot (tabular row) or a signal window | P(failure), predicted class, risk band | **AI4I 2020** (binary failure), **SECOM** (pass/fail), CWRU (multi-class fault diagnosis, wave 2), Paderborn (wave 2/3) |
| **B — RUL regression** | Regression of remaining useful life (cycles/minutes) from run-to-failure trajectories | RUL estimate, error vs true RUL | **C-MAPSS FD001** (existing), **FD003**, FD002/FD004 (wave 2), FEMTO-ST, XJTU-SY, IMS (wave 3) |
| **C — Anomaly detection** | One-class / unsupervised scoring of deviation from normal behaviour, evaluated against labels where they exist | Anomaly flag + score | **SECOM** (IsolationForest vs pass/fail labels), IMS (wave 3), AI4I secondary (train-on-normal) |

Rules that follow from the dataset research and are enforced by design:

- A dataset is only ever used for tasks its structure genuinely supports. AI4I rows
  are independent snapshots → no RUL. CWRU recordings are seeded-fault *diagnosis*,
  not future-failure prediction → the platform labels it Task A-diagnosis and any UI
  must use that wording. SECOM has no trajectories → no RUL.
- C-MAPSS subsets share sensors but **not** regimes: FD001/FD003 have a single
  operating condition; FD002/FD004 have six. (Note: FD003's distinguishing feature
  is its **two fault modes**, HPC + fan — not six conditions. Feature derivation for
  each subset is re-done from that subset's own data; see §3.)

## 2. Unified pipeline

```
 raw files ──► [Adapter] ──► canonical frame + metadata ──► [Validation]
                                                              │  schema, ranges,
                                                              │  duplicates, missingness report
                                                              ▼
 [Experiment store] ◄── [Evaluation] ◄── [Training] ◄── [Preprocessing]
        ▲                    ▲                 │           missing-value policy,
        │                    │                 │           scaling, encoding,
        └──── [XAI] ─────────┘                 │           feature derivation
                                               ▼
                                   task-specific view:
                                   A/C tabular matrix · B windowed sequences
```

- **Adapters** (`platform_lib/adapters/`): one class per dataset implementing
  `load() -> (pd.DataFrame, metadata: dict)`. The adapter is the *only* place that
  knows a dataset's file layout, column names, label encoding, and quirks
  (SECOM's `NaN` strings and separate labels file; AI4I's failure-mode columns;
  C-MAPSS's headerless 26-column text). Adapters never impute, scale, or split.
- **Validation**: row/column counts vs the research report's verified numbers,
  duplicate detection, missingness profile, label-rate sanity. A failed check
  aborts the experiment rather than silently training on corrupt data.
- **Preprocessing** is a *fitted, versioned* object per (dataset, task): every
  fitted statistic (imputation medians, dropped-column list, scaler mean/std,
  category maps, C-MAPSS feature derivation) is computed on the **training split
  only** and serialised into the experiment record. Test data never influences
  any fitted quantity — the same rule the FD001 pipeline already follows
  (scaler fit on train engines only; official test set untouched until scoring).
- **Task-specific views**: Task A/C consume a 2-D feature matrix. Task B consumes
  sliding windows `(N, seq_len, F)` built *inside* each asset (engine) — windows
  never cross asset boundaries — with the official C-MAPSS protocol (last window
  per test engine) for scoring.
- **Splitting protocols** (recorded verbatim in every experiment):
  - Tabular (AI4I, SECOM): stratified 80/20, seed 42.
  - C-MAPSS: engine-wise 80/20 train/val (seed 42, engines never on both sides)
    + the official provided test set for final scoring.
  - Signal datasets (later waves): split by recording/bearing, never by window —
    window-level random splits leak and inflate scores.

## 3. C-MAPSS feature derivation (per subset, from its own data)

The FD001 pipeline drops 7 of 21 sensors using FD001 statistics. That list is
**not** inherited by other subsets. For each subset the platform re-derives the
feature set from the subset's training split:

1. Compute per-column population std on the training split.
2. Drop sensor/setting columns whose std falls below a documented near-constant
   threshold (the FD001 precedent: sensor_6, std 0.00139, was dropped because
   z-scoring would amplify its noise ~700×).
3. Include the 3 operating settings as features (they are inputs a deployed
   system would know; for the single-condition subsets they carry little signal
   but cost nothing, and for FD002/FD004 they are essential regime context).
4. Record the kept/dropped lists and their stds in the experiment JSON.

## 4. Benchmarking engine

`platform_lib/benchmarking/run_benchmark.py` runs named experiment sets and
appends one JSON record per experiment to `platform/experiments/`, maintaining
`experiments/INDEX.json` (a machine-readable list of all records). No result is
ever reported anywhere except from these records.

### Experiment record schema

```jsonc
{
  "experiment_id":   "ai4i2020__taskA__random_forest__s42",
  "timestamp_utc":   "ISO-8601",
  "dataset":         "ai4i2020" | "secom" | "cmapss_fd003" | ...,
  "task":            "A" | "B" | "C",
  "model":           "random_forest",
  "model_params":    { ... },              // exact constructor args
  "seed":            42,
  "split": {
    "protocol":      "stratified 80/20, seed 42" | "engine-wise 80/20 + official test",
    "n_train": 0, "n_test": 0, "...": "task-specific counts (windows, engines, class balance)"
  },
  "preprocessing":   { ... },              // fitted artefacts: dropped cols, medians ref, features
  "metrics":         { ... },              // full metric set for the task (below)
  "timing": {
    "train_s":           0.0,
    "infer_ms_per_row":  0.0               // predict() wall time / n_test rows (or windows)
  },
  "environment": {
    "python": "...", "platform": "...",
    "libraries": { "numpy": "...", "pandas": "...", "scikit_learn": "...",
                   "torch": "...", "xgboost": "...", "lightgbm": "..." }
  },
  "git_commit":      "<local repo HEAD at run time>",
  "notes":           "free-text caveats (imbalance handling, leakage exclusions, ...)",
  "extra":           { }                   // e.g. permutation importance, feature derivation
}
```

### Metric sets (never accuracy alone)

- **Task A:** accuracy, precision, recall, F1, ROC-AUC, confusion matrix,
  plus the majority-class (all-negative) baseline accuracy for contrast —
  mandatory under imbalance (AI4I ≈3.4 % failures, SECOM ≈6.6 %).
- **Task B:** RMSE, MAE, NASA PHM08 score (asymmetric — over-predicting RUL is
  penalised harder), on the official test protocol; val metrics where a val
  split drives early stopping.
- **Task C:** precision, recall, F1 against held-out labels, false-alarm rate
  (fraction of normal rows flagged), contamination parameter stated explicitly.

Class imbalance is handled honestly: `class_weight="balanced"` (or equivalent)
where supported, recorded in `model_params`; raw and balanced variants are
distinguishable by experiment id.

## 5. Model-selection layer

When the platform later serves predictions (Phase 8), the selection layer obeys:

1. **Key = (dataset, task).** Candidate models are only ever compared *within*
   the same dataset and task. A ROC-AUC on AI4I and an RMSE on FD003 are never
   ranked against each other; there is no global "best model".
2. Within a key, rank by the task's primary metric (A: F1 — with recall reported
   alongside because missed failures are the expensive error; B: RMSE with the
   NASA score as tie-break/context; C: F1 at the stated contamination).
3. **Preprocessing travels with the model.** A model artefact is only loadable
   together with the exact fitted preprocessing record from its experiment JSON
   (feature list, order, medians, scaler). Mismatched preprocessing = hard error,
   never silent coercion — this is the platform form of the existing
   `preprocessing.json` Python↔Rust contract.
4. Every prediction response carries its experiment id, so any number shown in a
   UI is traceable to the record that produced the model.
5. Uncertainty is only surfaced where a method actually produces it
   (e.g. calibrated probabilities, MC-Dropout from the FD001 stretch work) —
   never invented confidence numbers.

## 6. Explainability (XAI) plan

- **Permutation importance** (model-agnostic, cheap): computed on the held-out
  test split for the strongest tabular model per dataset; stored in the
  experiment record's `extra`. Already the platform's working method on FD001.
- **Model-native importance**: impurity-based importances for forests/GBMs are
  recorded but labelled as biased toward high-cardinality features; they are
  never the sole explanation shown.
- **SHAP**: planned for the tree models (TreeExplainer is exact and fast) in a
  later phase; the experiment schema's `extra` block is where SHAP summaries land.
- **Integrated Gradients / permutation** for the LSTM (Task B) already exist in
  the FD001 system (`ml/src/predicai_ml/explain.py`) and remain the reference
  implementation for sequence models.
- Explanations may only reference features that exist: SECOM explanations name
  sensor *indices* (features are anonymised) — no physical story may be attached.

## 7. Maintenance history — optional input, never invented

None of the wave-1 datasets contains real maintenance records (AI4I's "tool wear"
is a process variable, not a service log; SECOM and C-MAPSS have no service data
at all). The platform therefore treats maintenance history as an **optional
side-input channel**:

- Adapters declare `metadata["maintenance_history"] = "absent"` when the dataset
  has none (AI4I, SECOM, C-MAPSS all do). The channel is then *structurally
  absent* from features — not zero-filled, not imputed, not synthesised.
- The existing FD001 system already implements the honest pattern: a SQLite
  maintenance log exists beside the model, its demo rows are explicitly flagged
  `is_synthetic = 1`, and maintenance data is **never** an LSTM input and
  servicing never resets predicted RUL. The platform keeps exactly that pattern.
- A controlled "does maintenance history help?" experiment is only meaningful on
  a dataset with real service logs; none of the 12 researched datasets provides
  one, so that experiment is documented as *not run — no suitable data*, rather
  than simulated with invented histories.

## 8. Where the existing FD001 system sits in the platform

| Existing component | Platform role |
|--------------------|---------------|
| `ml/` PyTorch LSTM + `preprocessing.json` contract | Registered entry **cmapss_fd001 × Task B**; its Kaggle-trained metrics (test RMSE 13.73 / MAE 9.89 / NASA 431.3, owner run) are the reference record the platform's local FD003/FD001 experiments compare against |
| ONNX export + parity tests | Serving format for sequence models; platform models must pass the same PyTorch↔ONNX parity bar before any serving claim |
| `edge/` Rust/Axum API | Serving layer for the FD001 entry; unchanged. Future platform serving (Phase 8) wraps the selection layer (§5) around per-key artefacts |
| `maintenance/` SQLite + policy | The maintenance-history channel of §7 (demo data flagged synthetic) |
| `simulator/` FD001 replay | Becomes the FD001 demo path; per-subset replay is a later phase |
| `dashboard/` React/TS | Phase 8 adds platform views (dataset explorer, benchmark tables from `experiments/INDEX.json`) — the dashboard reads experiment records, never hand-typed numbers |

## 9. Non-goals and honesty constraints (binding on all phases)

- No fabricated or placeholder metrics anywhere: dashboards and reports render
  only values present in experiment records.
- No claim of "12 datasets integrated": integration status is reported as the
  exact count of adapters with passing tests and at least one experiment record.
- No cross-dataset score comparisons (§5.1); cross-dataset *experiments*
  (Phase 6, e.g. train FD001 → test FD003) are designed separately with feature
  alignment stated up front, and their (expected) degradation is reported, not
  hidden.
- Simulated/synthetic data is always labelled as such (C-MAPSS = simulated;
  AI4I = synthetic; SECOM/CWRU = real).
