# PREDICAI Platform — Fault Diagnosis: what the platform can actually detect today

*Waves 5–6, 2026-11-15; RUL and bearing sections updated through
Wave 12 (Wave 13 sync, 2026-11-15). Every claim below is tied to an experiment
record in `platform/experiments/` (id given). If a capability is not
listed here with a record, the platform does not have it — regardless
of what any dataset could in principle support.*

## Summary

| Capability | Dataset | Best honest evidence | Record |
|---|---|---|---|
| Turbofan RUL (in-domain) | C-MAPSS FD001–FD004 | RMSE 13.17–15.37 (best per subset; per-regime normalisation on FD002/FD004) | see §1 |
| Bearing fault *type* (Normal / Inner / Outer / Ball), same rig & recordings distribution | CWRU | acc 0.802 / F1-macro 0.689 recording-wise (order-domain features, Wave 12) | `cwru__taskA__random_forest_order_recordingwise__s42` |
| Bearing fault *type*, same bearings, new measurements | Paderborn | acc 0.950 measurement-grouped (artificial); 0.9977 real-damage scope | `paderborn__taskA__random_forest__s42` |
| Bearing fault type across rigs/datasets | CWRU ↔ Paderborn | **fails** — three feature families (best acc 0.33 / 0.19; Normal F1 0.000) | §4 |
| Run-to-failure degradation alarm | IMS Test 1 | alarms only on the 2 bearings that actually failed, 337–723 snapshots early | `ims__taskC__isolation_forest__s42` |
| Machine failure (binary) | AI4I 2020 | F1 0.782 (LightGBM) | `ai4i2020__taskA__lightgbm__s42` |
| Process anomaly (weak) | SECOM | F1 0.244 (Isolation Forest) | `secom__taskC__isolation_forest__s42` |
| Battery RUL (cycles to 1.4 Ah EOL) | NASA Battery | LOBO RMSE 9.27 (RF) | `battery__taskB__random_forest_lobo__s42` |
| RUL across C-MAPSS subsets | FD001↔FD003, FD002↔FD004 | asymmetric: works wide→narrow, fails narrow→wide | §3 |

## 1. Turbofan degradation / RUL (C-MAPSS)

In-domain, per subset (official last-window protocol, cap 125):
FD001 — shipped LSTM RMSE **13.73** (`ml/artifacts/metrics.json`),
wave-4 GRU **13.22** (`cmapss_fd001__taskB__gru__s42`); FD002 —
per-regime LSTM **14.05** (`cmapss_fd002__taskB__lstm_perregime__s42`;
global-normalisation LSTM 16.24); FD003 — CNN-LSTM **13.17**
(`cmapss_fd003__taskB__cnn_lstm__s42`), LSTM 15.04; FD004 —
per-regime LSTM **15.37** (`cmapss_fd004__taskB__lstm_perregime__s42`;
global-normalisation LSTM 18.81). On the two 6-condition subsets,
per-regime normalisation (KMeans k=6 regimes fitted on train
settings, per-regime sensor z-scores) is worth −13.5% (FD002) and
−18.3% (FD004) test RMSE for the LSTM, and on FD004 it also retakes
the NASA-score lead from Ridge (1457 vs 1766) that the
global-normalisation LSTM had lost (2141). Claim scoped to seed 42.
What this supports: estimating
remaining cycles for engines from the *same* subset distribution.
What it does not support: any claim about a specific physical
component failing — C-MAPSS labels are RUL only; the two fault modes
(HPC degradation, fan degradation) are properties of the simulation
setup, and no record in this platform classifies them.

## 2. Bearing fault type (CWRU, Paderborn)

**CWRU** (12 recordings, 4 classes incl. Ball): the current best
honest result is the wave-12 order-domain record — leave-one-
recording-out Random Forest accuracy **0.802**, F1-macro **0.689**
(`cwru__taskA__random_forest_order_recordingwise__s42`), under the
identical protocol as the earlier feature families: time-domain
**0.665** / 0.421 (wave 4, `cwru__taskA__random_forest_recordingwise__s42`)
and spectral **0.679** / 0.417 (wave 9). The order-domain features
(characteristic defect frequencies expressed in shaft-speed-relative
units) are what finally make fault *location* partially readable
in-domain: InnerRace goes from 0/177 correct (time-domain) and
15/177 (spectral) to **109/177**. The wave-2 window-split figure of
1.000 (`cwru__taskA__random_forest__s42`) is within-recording
separability and must never be quoted as generalisation. Even
recording-wise, all recordings come from one rig and one bearing
type, and fault *location* is what fails to transfer across rigs
(§4).

**Paderborn** (3 bearings, 3 classes): measurement-grouped RF
accuracy **0.950** (`paderborn__taskA__random_forest__s42`).
Permutation importance on that split (`paderborn__taskA__permutation_rf__s42`)
ranks **kurtosis (0.196)** and **skewness (0.164)** far above the
amplitude features (std 0.038, RMS 0.038) — the model keys on
impulsiveness, which is physically consistent with localised race
defects, but it is still one physical bearing per class: unseen
bearings are unproven. Wave 10 added a second scope — *real*
fatigue-damage bearings KA04 (outer race) and KI04 (inner race),
damage verified from the official per-bearing damage-profile PDFs:
grouped RF accuracy **0.9977** (`paderborn_real__taskA__random_forest__s42`),
under the same one-bearing-per-class caveat; KI04 also carries a
documented secondary outer-race deformation, stated in its record.

## 3. Cross-subset RUL transfer (C-MAPSS)

| Direction | Model | Transfer RMSE | Target in-domain | Verdict |
|---|---|---|---|---|
| FD001 → FD003 | shipped LSTM | **48.80** | 15.04 (LSTM) | fails |
| FD003 → FD001 | CNN-LSTM | **15.82** | 13.73 (shipped LSTM) | close to in-domain |
| FD002 → FD004 | Ridge | **51.38** | 19.43 (Ridge) | fails |
| FD004 → FD002 | Ridge | **18.07** | 16.58 (Ridge) | mild degradation |

Records: `cmapss_fd001_to_fd003__taskB__transfer_lstm__s42`,
`cmapss_fd003_to_fd001__taskB__transfer_cnn_lstm__s42`,
`cmapss_fd002_to_fd004__taskB__transfer_ridge__s42`,
`cmapss_fd004_to_fd002__taskB__transfer_ridge__s42`.

The pattern is consistent across both pairs and both model families:
transfer **from the wider domain to the narrower one works**
(FD003's two fault modes → FD001's one; FD004's two modes → FD002's
one), and transfer **into a domain containing an unseen fault mode
fails** — in the narrow→wide direction the models systematically
read the unseen mode's engines as healthy (FD002→FD004: 62 of 237
engines over-predicted by >50 cycles, predictions piling at the
125-cycle cap region while true RUL is 7–24). A model that has never
seen a failure mode cannot be trusted to price it. Note the pairs
are not perfectly controlled experiments (the FD003→FD001 direction
also changes architecture, LSTM→CNN-LSTM, and feature set,
14→15 with sensor_6) — the report section states this.

## 4. Cross-dataset bearing transfer (CWRU ↔ Paderborn) — fails

Records — identical protocol in all three waves, the feature set
the only variable. Time-domain (wave 5):
`cwru_to_paderborn__taskA__transfer_rf__s42` (accuracy **0.253**,
F1-macro 0.205) and `paderborn_to_cwru__taskA__transfer_rf__s42`
(**0.125**, 0.156). Spectral (wave 9):
`cwru_to_paderborn__taskA__transfer_rf_spectral__s42` (**0.210**)
and `paderborn_to_cwru__taskA__transfer_rf_spectral__s42`
(**0.187**). Order-domain (wave 12):
`cwru_to_paderborn__taskA__transfer_rf_order__s42` (**0.335**)
and `paderborn_to_cwru__taskA__transfer_rf_order__s42` (**0.163**)
(F1-macro 0.169 / 0.206). Against
in-domain references of 0.802 (CWRU recording-wise, order features)
and 0.950 (Paderborn grouped), every direction of every family is
at or below the target's majority-class rate. The sharpest finding
is unchanged across all three families: **the Normal class does
not transfer** — its F1 is 0.0 in both directions for the
time-domain and order-domain families (a CWRU-trained forest
labels every Paderborn window Inner/Outer; a Paderborn-trained
forest labels all 591 CWRU Normal windows OuterRace); the spectral
family reached 0.195 in one direction only (CWRU→Paderborn) while
collapsing elsewhere. With identical features and label names,
what differs is everything else: 12 kHz vs 64 kHz sampling (a
2048-sample window spans 170.7 ms vs 32.0 ms), rigs, bearings,
speeds, loads, defect creation. Conclusion: with hand-crafted
features — time-domain, spectral, or order-domain — "Normal" is a
property of a rig, not of a bearing. The order features prove the
point by contrast: they transform in-domain diagnosis (§2)
precisely because they encode one rig's physics, and that is
exactly what does not cross to another rig. Fault-location
diagnosis does **not** transfer across these datasets, and no
platform output may suggest otherwise. The cross-rig question is
closed negatively for hand-crafted feature families; learned
representations are a different, untested question.

## 5. Run-to-failure anomaly alarming (IMS Test 1)

Per-bearing Isolation Forests fitted on the first 30% of the run
(`ims__taskC__isolation_forest__s42`; threshold = train q95, alarm =
10 consecutive snapshots above it):

| Bearing | Teardown truth | First sustained alarm | Last-10% exceedance |
|---|---|---|---|
| 1 | no defect reported | **never** | 0.10 |
| 2 | no defect reported | **never** | 0.19 |
| 3 | inner-race defect | snapshot **1819 / 2156** (84% of run; 337 snapshots ≈ 2.3 days early) | 1.00 |
| 4 | roller-element defect | snapshot **1433 / 2156** (66% of run; 723 snapshots ≈ 5.0 days early) | 1.00 |

This is the platform's cleanest fault-detection result: the
detectors fire on exactly the bearings that failed, and only there.
Its limits are equally clear: one run, one rig, four bearings on a
shared shaft; alarm timing is not a diagnosis (the detector cannot
say *which* component failed — bearings 3 vs 4 are separated only
because each has its own channels); and there are no per-snapshot
labels, so no precision/recall exists for this dataset.

## 6. Machine failure, binary (AI4I 2020)

LightGBM F1 **0.782**, ROC-AUC 0.970 (`ai4i2020__taskA__lightgbm__s42`);
SHAP and permutation agree on the drivers (Torque, Rotational speed,
temperatures, Tool wear — `ai4i2020__taskA__shap_lightgbm__s42`).
Limits: AI4I is *synthetic* data generated from known failure rules,
and the platform models predict the binary failure flag only — the
dataset's failure-type columns (TWF/HDF/PWF/OSF/RNF) are **not**
modelled, so the platform cannot name an AI4I failure type.

## 7. Manufacturing process anomaly (SECOM) — weak signal

Best Task-C result is the wave-1 Isolation Forest, F1 **0.244**
(`secom__taskC__isolation_forest__s42`); wave 4's One-Class SVM
(0.111) and autoencoder (0.049) are worse. SECOM has no fault-type
labels (pass/fail only, 104 fails of 1,567), so nothing about *what*
failed can be said. Treat SECOM outputs as weak triage signals, not
diagnoses.

## 8. Battery degradation / RUL (NASA Li-ion)

Leave-one-battery-out: RF pooled RMSE **9.27 cycles**
(`battery__taskB__random_forest_lobo__s42`) vs LSTM 14.92 — with
three batteries, per-battery spread is large and the pooled number
is a small-sample estimate. SHAP on the all-batteries forest
(`battery__taskB__shap_random_forest__s42`) ranks **soh (16.0)** and
**capacity_ah (15.1)** — two views of the same measured capacity
trajectory — far above everything else (discharge duration 7.3;
all voltage/temperature features ≤ 1.4; ambient temperature 0.0).
Reading: the model's RUL is essentially a capacity-curve
extrapolation. That is legitimate for this dataset, but it means
the model has nothing to say about *why* a cell is ageing, and
B0007 (censored, never reached EOL) is excluded from every battery
score in the platform.

## 9. Hard limits — what the platform must never claim

1. **No component-level electrical diagnosis.** No integrated
   dataset carries wire/component-level electrical labels; the
   Paderborn current channels are not featurised, and no motor
   electrical-fault dataset has been integrated (candidates are
   researched-only in DATASET_REGISTRY.md). Any statement like
   "this wire/phase/winding is defective" is unsupported by every
   record in this platform.
2. **No cross-rig fault-location claims** (§4): bearing fault type
   does not transfer between CWRU and Paderborn with any of the
   three hand-crafted feature families tested (time-domain,
   spectral, order-domain).
3. **No transfer into unseen fault modes** (§3): RUL models price
   unseen failure modes as health. Zero-shot deployment across
   subsets/datasets is not supported by any result here.
4. **Explanations are not mechanisms.** SHAP/permutation/IG
   rankings describe what a model uses (e.g. battery capacity
   features, Paderborn kurtosis), on that model's training
   distribution. They are not evidence of physical causation.
5. **Synthetic and seeded data are labelled as such.** AI4I is
   synthetic; CWRU faults and the original Paderborn scope are
   EDM-seeded; C-MAPSS is simulated. The naturally-developed
   failures are the IMS and battery run-to-failure recordings
   plus the wave-10 Paderborn fatigue bearings (KA04/KI04) —
   all single-rig studies.
6. **Maintenance history is absent** from every integrated dataset;
   the platform never invents it (ARCHITECTURE.md §6).
