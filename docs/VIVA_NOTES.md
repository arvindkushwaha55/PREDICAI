# PREDICAI — Viva / Interview Notes

Short, honest answers. Everything below is true of the shipped system;
anything marked TODO is genuinely not done.

**Q: What does PREDICAI predict, and in what unit?**
The Remaining Useful Life (RUL) of a turbofan engine, in **cycles** —
the unit of the NASA C-MAPSS data. Never hours: converting cycles to
hours would need a duty-cycle assumption we do not have.

**Q: Why regression, not classification?**
"How much life is left" is a number, not a category. Maintenance planning
needs the number; alert bands can always be derived from it (our policy
layer does exactly that).

**Q: Why cap RUL at 125?**
Standard C-MAPSS practice: a healthy engine's sensors carry almost no
degradation signal, so uncapped labels force the model to fit noise in
the long flat region. 125 is a literature convention (Heimes-style
piecewise-linear labels), not a proven optimum — the ablation
(125 vs 150 vs no cap) is the planned check.

**Q: Why did you drop 7 sensors?**
Six sensors are *exactly constant* on FD001 (zero variance — they carry
no information), plus `sensor_6`, whose standard deviation (0.0014) is so
small that z-score normalization would amplify its noise ~700×.
14 sensors remain. Settings were dropped for the same reason
(`setting_3` is constant on FD001).

**Q: How do you know there's no data leakage?**
Three guards: the train/validation split is by *engine*, never by window
(no engine appears in both); the scaler is fit on the training split only;
and the FD001 test protocol uses one window per engine (its last 30
cycles) with official RUL labels. The Kaggle run's preprocessing artifact
was verified byte-identical to an independent local run.

**Q: Your headline numbers?**
Test (last window per engine, n=100): **RMSE 13.73 cycles, MAE 9.89,
NASA PHM08 score 431.3**. Validation (all windows, n=3,272): RMSE 13.01.
Published FD001 LSTM results sit around 12.5–16.5 RMSE, so this is a
solid, believable baseline — not a record claim. Error is smallest near
failure (true RUL 0–25: MAE 2.92), which is where accuracy matters.

**Q: Why an LSTM?**
RUL depends on degradation *history*, not a single snapshot; the input is
a 30-cycle sequence. LSTMs are the standard, well-understood baseline for
this benchmark. CNN/Transformer comparisons are future work, not claimed.

**Q: What does the NASA score penalize?**
Late predictions (predicting more life than remains) more than early
ones — an asymmetric exponential penalty, because overestimating life is
the dangerous direction operationally.

**Q: Explainability — which sensors matter?**
Two methods, reported with their disagreement: permutation importance
puts sensor_14, sensor_9, sensor_2 on top; Integrated Gradients puts
sensor_11, sensor_12, sensor_8 on top (Spearman ρ = 0.56 between the
rankings). Both agree sensor_11/sensor_4 matter and sensor_3/sensor_17
matter least. Importance means the model *relies on* a sensor —
association, not proof that the sensor causes failure.

**Q: Why Rust for the API if the model is Python?**
Deployment realism: ONNX is the contract between the two stacks, so the
serving layer can be a single small native binary at the edge. We make
**no speed claim** beyond the benchmark actually measured on this
machine (see README, measured over 1,000 local requests).

**Q: How do you know the Rust service runs the same model?**
Parity is tested, not assumed: PyTorch ↔ ONNX max difference 1.53e-05;
the Rust API's prediction for test engine 1 (117.60932) matches the
Kaggle PyTorch prediction to 5 decimal places; an automated Rust test
re-checks parity against Python fixtures on every run.

**Q: Is the maintenance history real?**
No — the seeded service records are **synthetic demo data**, flagged
`is_synthetic` in the database and labelled in the UI. They demonstrate
the maintenance workflow; they are never fed to the model, and servicing
never "resets" the model's RUL — history and prediction stay separate.

**Q: Are the alert thresholds (30/50 cycles) scientifically validated?**
No, and we say so in the product: they are conservative *policy choices*
for when a human should look, not failure boundaries.

**Q: What are the honest limitations?**
One dataset subset (FD001, single operating condition, sea level); no
real sensors — the simulator replays recorded test data; ONNX Runtime is
loaded dynamically from the Python package in this build; the `ort`
crate version is a pinned release candidate (no stable 2.0 exists yet);
dashboard visuals were build-verified, and its live panel needs the API
running locally. TODO: baselines + cap ablation numbers, FD002–FD004,
uncertainty estimates.
