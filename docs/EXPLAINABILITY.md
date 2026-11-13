# PREDICAI — Explainability (Part 6 / Phase A5)

**Question answered:** which of the 14 sensors does the trained LSTM
*rely on* when it predicts Remaining Useful Life — and do two independent
methods agree?

**Language rule (binding):** everything below describes **association** —
which sensors move the model's prediction — never causation. No result
here shows that a sensor *causes* degradation or failure.

- Model: `ml/artifacts/best_model.pt` (Kaggle-trained LSTM, test RMSE
  13.73 cycles — see `ml/artifacts/metrics.json`)
- Data: validation split (engine-wise holdout), the split the model never
  trained on. Permutation uses all **3,272** val windows; Integrated
  Gradients a deterministic subsample of **200** (seed 123).
- Code: `ml/src/predicai_ml/explain.py`, run by
  `ml/scripts/run_explainability.py`. Full machine-readable results:
  `ml/artifacts/explainability.json`. Charts:
  `permutation_importance.png`, `integrated_gradients.png`.
- SHAP is **not** used: on raw LSTM sequences it is methodologically
  weak (breaks temporal dependence, unstable rankings on C-MAPSS) —
  see RESEARCH_REPORT.md §9. It stays secondary/experimental by design.

## Method 1 — Permutation importance (model-agnostic)

Baseline RMSE on the val windows, then one sensor at a time has its
whole 30-cycle trace shuffled across windows (seed 123, 1 repeat). The
shuffle keeps the sensor's marginal distribution but destroys its
alignment with each window, so the RMSE increase measures how much
accuracy the model loses without that sensor's per-window information.

## Method 2 — Integrated Gradients (manual, no Captum)

Attributions along the straight-line path from a **zero baseline** to
each window, 50 steps (midpoint rule), gradients of the raw model
output, averaged over the 200 sampled windows. Inputs are z-scored, so
the zero baseline is the **train-mean operating point** — the "average
engine". The model's prediction at that baseline is **113.72 cycles**
(near the 125 cap: average-looking readings read as a healthy engine).
Attributions are in cycles of predicted RUL: per window they sum to
`prediction − 113.72` (completeness, verified below).

## Results

### Permutation importance (baseline val RMSE = 13.013 cycles)

| # | Sensor | Permuted RMSE | RMSE increase |
|---|---|---|---|
| 1 | sensor_14 | 18.191 | **+5.178** |
| 2 | sensor_9 | 17.596 | **+4.583** |
| 3 | sensor_2 | 15.850 | +2.836 |
| 4 | sensor_11 | 15.674 | +2.660 |
| 5 | sensor_4 | 15.494 | +2.481 |
| 6 | sensor_8 | 15.434 | +2.420 |
| 7 | sensor_12 | 14.324 | +1.310 |
| 8 | sensor_7 | 14.230 | +1.217 |
| 9 | sensor_13 | 14.028 | +1.015 |
| 10 | sensor_21 | 13.793 | +0.780 |
| 11 | sensor_20 | 13.720 | +0.707 |
| 12 | sensor_15 | 13.695 | +0.682 |
| 13 | sensor_3 | 13.537 | +0.524 |
| 14 | sensor_17 | 13.457 | +0.444 |

### Integrated Gradients (mean |attribution| per sensor, cycles)

| # | Sensor | Mean \|attribution\| | Mean signed |
|---|---|---|---|
| 1 | sensor_11 | **0.5331** | −0.1118 |
| 2 | sensor_12 | **0.5319** | −0.0876 |
| 3 | sensor_8 | 0.5059 | −0.1079 |
| 4 | sensor_7 | 0.5002 | −0.0931 |
| 5 | sensor_4 | 0.4876 | −0.0574 |
| 6 | sensor_13 | 0.4789 | −0.1039 |
| 7 | sensor_2 | 0.4186 | −0.1052 |
| 8 | sensor_9 | 0.4137 | −0.0815 |
| 9 | sensor_14 | 0.3953 | −0.0812 |
| 10 | sensor_20 | 0.3073 | −0.0640 |
| 11 | sensor_15 | 0.3030 | −0.0381 |
| 12 | sensor_17 | 0.2695 | −0.0058 |
| 13 | sensor_21 | 0.2685 | +0.0115 |
| 14 | sensor_3 | 0.2315 | −0.0436 |

Signed means are negative for 13 of 14 sensors: relative to the
113.72-cycle "average engine" baseline, real val windows mostly push
the prediction *down* (mean prediction on the IG sample: 84.64 cycles).
sensor_21 is the lone exception, and its signed mean (+0.0115) is near
zero. Per-timestep, attribution mass grows toward the end of the window
(mean |attribution| 0.18 at cycle 1 → ~0.98 at cycle 29 of 30) — the
model leans most on the most recent cycles, as expected for an LSTM
whose head reads the last hidden state.

## Agreement and disagreement (Phase A5 gate)

- **Overall: moderate agreement.** Spearman rank correlation between
  the two full rankings is **ρ = 0.56** (p = 0.037).
- **Agreed top sensors:** sensor_11 and sensor_4 are top-5 in **both**
  methods; sensor_2 and sensor_8 are in both top-7s. This is consistent
  with the FD001 literature, where the temperature/pressure sensors
  around the high-pressure compressor (the FD001 fault mode is HPC
  degradation) are repeatedly reported to trend with degradation.
- **Agreed bottom:** sensor_3 and sensor_17 sit in the bottom three of
  **both** rankings.
- **Disagreement at the very top (reported as-is):** permutation ranks
  sensor_14 and sensor_9 #1–#2, but IG places them only #8–#9; IG's
  leaders sensor_12 and sensor_7 are mid-table (#7–#8) for permutation.
  The methods measure different things — permutation measures reliance
  for *accuracy*, IG measures *attribution mass* along one path — so
  exact agreement is not expected, and the gate's fallback (document
  the disagreement honestly) applies to the #1 spot specifically.
- **Two honest surprises, flagged rather than explained away:**
  1. sensor_9 / sensor_14 top the permutation ranking although they
     are often reported as weakly trending or noisy *univariately* in
     FD001. An LSTM can still exploit them through multivariate /
     temporal patterns — or shuffling them disrupts correlated
     structure the model leans on. Either way: association, not a
     degradation mechanism.
  2. sensor_3 (HPC outlet temperature) is near the bottom of both
     rankings although it is one of the most strongly trending sensors
     in the literature. The likely statistical reason: it is highly
     correlated with sensor_2 / sensor_4, so the model can compensate
     when any single one is shuffled, and IG splits credit across the
     correlated block. Importance here is *marginal reliance of this
     model*, not physical relevance.

## Sanity checks (all run, all recorded in the JSON)

| Check | Result |
|---|---|
| Permutation baseline vs metrics.json val RMSE (13.013209) | 13.013211 — diff 0.000002 ✓ |
| IG completeness: Σ attributions vs prediction − baseline prediction | mean abs error 0.0039 cycles; mean relative error **0.0115 %** ✓ |
| All 14 sensors in both rankings | yes ✓ |
| NaN / Inf anywhere | none ✓ |
| Unit tests (`ml/tests/test_explain.py`) | 3/3 pass ✓ |

## Limitations

- **Correlation is not causation.** These are statements about a
  trained network's behaviour on one validation split, not about engine
  physics. A sensor can be physically central and score low (sensor_3,
  above) or score high for purely statistical reasons.
- **Correlated sensors share and shift credit.** Permutation
  under-states a sensor whose information duplicates another's; IG
  distributes credit along the path. Grouped/block permutation would
  be the follow-up if per-sensor credit within the temperature block
  (2 / 3 / 4) ever needs separating.
- **The zero baseline is a modelling choice.** Zero in z-score space =
  the train-mean operating point, whose prediction (113.72) sits near
  the RUL cap; attributions therefore describe "deviation from an
  average engine", and a different baseline (e.g. an end-of-life
  reference) would give a different — also valid — view.
- **Permutation used 1 repeat.** With 3,272 windows the shuffle noise
  is small, but the closest mid-table gaps (~0.03 cycles between
  sensor_20 / sensor_15) are within plausible repeat-to-repeat noise —
  do not over-read adjacent mid-table ranks.
- **IG averaged over 200 windows.** Per-engine or per-RUL-band
  attributions can differ; the aggregate hides that variation.
- Results are for **this** model, seed and split. Retraining can move
  mid-table ranks; the top/bottom pattern is the robust part.
