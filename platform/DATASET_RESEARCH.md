# PREDICAI Platform — Dataset Research Report

**Original research:** 2026-11-13 (Phase 1 of the platform expansion)
**This version:** full rewrite, 2026-11-15 (advanced expansion, Wave 8)

**Why a rewrite:** the Phase-1 file was saved truncated — it ended
mid-§10 (Milling) with a literal `...[truncated 9113 chars]` marker, and
its §§11–13, first-wave shortlist and "could not verify" list survived
only in the research handoff, not on disk. A wave 1–3 addendum was later
appended to the truncated file. This rewrite restores one complete,
coherent document from the current sources of truth:
`DATASET_REGISTRY.md` (the operational registry, 20 entries, kept
current with integration reality), the intact Phase-1 sections, and the
wave 1–3 addendum — updated for everything Waves 1–7 changed (Paderborn
licence confirmed and integrated, NASA Battery integrated, IMS
integrated in Wave 5 after the user approved the 1.08 GB download,
MFPT found blocked). Where this report and the registry ever disagree,
the registry wins — it is the living document.

**Scope:** multi-dataset, multi-model predictive-maintenance platform.
Tasks: **A** = failure/fault classification · **B** = RUL regression ·
**C** = anomaly detection.

**Method:** every factual claim was checked against a live source
(official repository page, source-organisation page, or the dataset's
own documentation) during the research sessions of 2026-11-13 and
2026-11-15. The source used is recorded per dataset. Licence entries
quote what the official source states; "not confirmed" / "not stated"
means no licence text was found on the official page — it does **not**
mean the data is public domain. Anything that could not be confirmed is
explicitly flagged, and collected again in the final "Not verified"
section. No dataset was invented to fill a category: where no verified
public dataset exists (automotive engine health), the category is
recorded as unfilled.

**Result:** 20 registry entries — 18 distinct datasets, one unfilled
category (automotive), and one candidate group (motor electrical
faults). The PHM 2008 Challenge data is the same family as C-MAPSS and
is documented under §1 rather than counted separately. As of this
rewrite, **10 entries are integrated** (adapter + local data + at least
one experiment record, or served by the original PREDICAI system),
**2 are blocked** (Bosch, MFPT), and the rest are researched-only.

---

## Summary comparison table

| # | Dataset | A | B | C | Licence (as stated at source) | Status |
|---|---------|:---:|:---:|:---:|---|---|
| 1 | C-MAPSS FD001 | (derived) | ✅ | (derived) | NASA research use; formal text not confirmed | **integrated** (original system + platform) |
| 2 | C-MAPSS FD002 | (derived) | ✅ | (derived) | NASA research use; formal text not confirmed | **integrated** |
| 3 | C-MAPSS FD003 | (derived) | ✅ | (derived) | NASA research use; formal text not confirmed | **integrated** |
| 4 | C-MAPSS FD004 | (derived) | ✅ | (derived) | NASA research use; formal text not confirmed | **integrated** |
| 5 | AI4I 2020 | ✅ | ❌ | ✅ | CC BY 4.0 | **integrated** |
| 6 | SECOM | ✅ | ❌ | ✅ | CC BY 4.0 | **integrated** |
| 7 | CWRU Bearing | ✅ | ❌ | ✅ | Not stated on official page | **integrated** (4 recordings) |
| 8 | Paderborn Bearing | ✅ | ❌ | ✅ | **CC BY-NC 4.0** (confirmed on official page) | **integrated** (3 of 32 bearings) |
| 9 | IMS Bearing | ❌ | (derived) | ✅ | Not stated on PCoE page | **integrated** (Test 1) |
| 10 | NASA Li-ion Battery | ❌ | ✅ | ❌ | No named licence; acknowledgement requested | **integrated** (4 batteries) |
| 11 | FEMTO-ST PRONOSTIA | ❌ | ✅ | ✅ | Not confirmed | researched-only |
| 12 | XJTU-SY Bearing | ❌ | ✅ | ✅ | Not stated | researched-only |
| 13 | Milling Tool Wear | ✅ | ✅ | ✅ | NASA portal; formal text not confirmed | researched-only |
| 14 | Hydraulic Condition Monitoring | ✅ | ❌ | ✅ | CC BY 4.0 | researched-only |
| 15 | MetroPT-3 | (derived) | ❌ | ✅ | CC BY 4.0 | researched-only |
| 16 | Bosch Production Line | ✅ | ❌ | ✅ | Kaggle competition rules | **blocked** — Kaggle-gated |
| 17 | MIMII Sound | (derived) | ❌ | ✅ | CC BY-SA 4.0 | researched-only (by decision — 100.2 GB) |
| 18 | MFPT Bearing | ✅ | ❌ | ❌ | Not stated (live page gone) | **blocked** — official site retired |
| 19 | Automotive engine health | — | — | — | — | **no verified public sensor dataset found** |
| 20 | Motor electrical faults (candidates) | ✅ | ❌ | ❌ | CC BY 4.0 / CC0 (verified per candidate) | researched-only |

---

## 1. C-MAPSS FD001 — NASA Turbofan Engine Degradation Simulation

- **Source:** NASA Ames Prognostics Center of Excellence (PCoE),
  Prognostics Data Repository —
  https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/
  Simulated with the C-MAPSS tool (Saxena & Goebel, 2008). Downloaded
  via the PHM Society S3 mirror of the repository archive
  (`6. Turbofan Engine Degradation Simulation Data Set.zip`). The
  PHM 2008 Challenge data is this same data family and is not counted
  as a separate dataset.
- **Licence:** US Government (NASA) work, distributed for research use
  with citation of Saxena et al. (2008); a formal licence text on the
  download page was **not confirmed**.
- **Tasks:** B (primary). A/C only in derived form (RUL risk bands,
  residuals) — there is no native per-cycle failure label.
- **Features / labels:** per cycle — unit, cycle, 3 operating settings,
  21 sensors. Label = RUL in cycles (train run-to-failure; test
  truncated, true RUL in `RUL_FD001.txt`).
- **Size (verified):** train 20,631 rows / 100 engines; test 13,096
  rows / 100 engines. 1 operating condition, 1 fault mode (HPC
  degradation).
- **Preprocessing:** drop constant/near-constant sensors (FD001 keeps
  14 of 21 sensors; the 3 settings are constant and dropped); z-score
  with train-engine statistics only; windows of 30 cycles; RUL capped
  at 125.
- **Limitations:** simulated, not real engines; single asset type.
- **Status:** **integrated** — the original PREDICAI system (LSTM →
  ONNX → Rust API → dashboard) plus the platform `CMapssAdapter`.
  Maintenance history: absent (never invented).

## 2. C-MAPSS FD002

- **Source / licence:** as §1.
- **Tasks:** B.
- **Features / labels:** same 26-column schema; 6 operating conditions,
  1 fault mode (HPC degradation).
- **Size (verified):** train 53,759 rows / 260 engines; test 33,991
  rows / 259 engines.
- **Preprocessing:** keeps all 3 settings + 20 sensors (23 features —
  only sensor_16 drops); the settings genuinely vary across the 6
  conditions, so they are features, not constants. Per-regime
  normalisation remains an open improvement (noted since wave 2).
- **Status:** **integrated** — `CMapssAdapter(subset="FD002")`; Ridge,
  LSTM, Random Forest and XGBoost records exist.

## 3. C-MAPSS FD003

- **Source / licence:** as §1.
- **Tasks:** B.
- **Features / labels:** same schema; 1 operating condition, 2 fault
  modes (HPC + fan degradation).
- **Size (verified):** train 24,720 rows / 100 engines; test 16,596
  rows / 100 engines.
- **Preprocessing:** drops all 3 settings (constant in this
  single-condition subset) + 6 constant sensors, and *keeps* sensor_6
  → 15 features (contrast FD001's 14).
- **Status:** **integrated** — `CMapssAdapter(subset="FD003")`; Ridge,
  LSTM and CNN-LSTM records exist, and FD003 is both target (of the
  FD001→FD003 zero-shot transfer) and source (FD003→FD001) in the
  transfer matrix.

## 4. C-MAPSS FD004

- **Source / licence:** as §1.
- **Tasks:** B.
- **Features / labels:** same schema; 6 operating conditions, 2 fault
  modes (HPC + fan degradation).
- **Size (verified):** train 61,249 rows / 249 engines; test 41,214
  rows / 248 engines.
- **Preprocessing:** as FD002 (23 features; only sensor_16 drops).
- **Status:** **integrated** — `CMapssAdapter(subset="FD004")`; Ridge
  + LSTM records exist. Note: on FD004 the LSTM wins RMSE/MAE but
  loses the NASA score to Ridge — recorded as measured.

## 5. AI4I 2020 Predictive Maintenance Dataset

- **Source:** UCI Machine Learning Repository —
  https://archive.ics.uci.edu/dataset/601/predictive%2Bmaintenance%2Bdata
  (DOI 10.24432/C5HS5C). Donated by Matzka (2020); a **synthetic**
  dataset modelled on a real milling-machine process.
- **Licence:** CC BY 4.0.
- **Tasks:** A (primary — the cleanest classification-native dataset in
  this list), C (secondary — train on non-failure rows). Not B: rows
  are independent process snapshots, not run-to-failure trajectories,
  so no RUL exists.
- **Features / labels:** 10,000 rows × 14 columns — air/process
  temperature, rotational speed, torque, tool wear, product type;
  binary `Machine failure` label (339 failures, ≈3.4%) plus 5
  failure-mode flags (TWF/HDF/PWF/OSF/RNF).
- **Preprocessing:** the 5 mode flags *define* the target and are
  declared leakage columns (never features); UDI/Product ID are
  identifiers; strong imbalance → PR-oriented metrics and class
  weighting. The failure rules are deterministic functions of the
  features, so very high scores are achievable and must not be
  over-interpreted as real-world performance.
- **Status:** **integrated** — `AI4IAdapter`; classifier records
  (LogReg, RF, HGB, XGBoost, LightGBM, SVM, MLP) + permutation
  importance + SHAP record exist. Every downstream claim must keep the
  synthetic caveat.

## 6. SECOM Manufacturing Dataset

- **Source:** UCI Machine Learning Repository —
  https://archive.ics.uci.edu/dataset/179/secom (DOI 10.24432/C54305).
  Semiconductor manufacturing process data (McCann & Johnston, 2008).
- **Licence:** CC BY 4.0.
- **Tasks:** A and C (both primary fits — only 104 labelled failures
  makes supervised learning fragile, and one-class approaches on the
  pass class are a standard, defensible treatment). Not B.
- **Features / labels:** 1,567 rows × 590 anonymised sensor columns
  (the UCI prose says 591 features; the released matrix has 590
  measurement columns + label — a known off-by-one in the repository
  text); pass (−1) / fail (+1) → platform target fail = 1 (104 fails,
  ≈6.6%).
- **Preprocessing:** extensive missingness (≈4.5% of cells; some
  features mostly missing) → missingness filtering + imputation +
  constant-feature drop (wave-1 pipeline: 590 → 450 surviving
  features, fitted on train only). Features are anonymised —
  explanations may cite sensor indices only, never physical names.
- **Limitations:** small n against ~590 features (overfitting risk
  dominates); anonymisation blocks physical interpretation and any
  cross-dataset feature alignment; 2008 process data.
- **Status:** **integrated** — `SECOMAdapter`; LR/RF classification
  records + Isolation Forest, One-Class SVM and autoencoder anomaly
  records exist.

## 7. CWRU Bearing Data (Case Western Reserve University)

- **Source:** CWRU Bearing Data Center —
  https://engineering.case.edu/bearingdatacenter/download-data-file
  Recordings are served as individual `.mat` files from the university
  host (no account). Community mirrors exist but the university site
  is the canonical source.
- **Licence:** **not stated** on the official download pages (verified:
  the pages present data tables without a licence statement; academic
  use with citation is the de-facto norm — flag before any commercial
  use).
- **Tasks:** A (primary — fault *diagnosis*: this detects an existing
  seeded fault, it does not predict a future failure; wording matters).
  C secondary (train on Normal only). Not B: no run-to-failure
  progression.
- **Features / labels:** drive-end vibration at 12 kHz (also 48 kHz
  and fan-end data exist); classes Normal / InnerRace / Ball /
  OuterRace, crossed with fault diameter (0.007″–0.028″) and load
  (0–3 HP). Bearings: SKF 6205-2RS JEM (drive end). Integrated scope:
  exactly 4 recordings at 0 HP with 0.007″ faults — 97.mat Normal
  (243,938 samples), 105.mat IR007_0 (121,265), 118.mat B007_0
  (122,571), 130.mat OR007@6_0 (121,991); identity self-verified via
  the `X<id>_DE_time` variables. Wave 4 added further recordings for
  the recording-wise re-evaluation.
- **Preprocessing:** non-overlapping 2,048-sample windows → 10
  time-domain features per window (296 windows in the 4-recording
  scope). **Binding caveat:** one recording per class in the original
  scope, so window-wise splits share recordings across train/test —
  scores measure within-recording separability, not generalisation.
  The wave-4 recording-wise re-evaluation (accuracy 0.665 vs the
  window-split 1.000) is the honest headline; see
  `EXPERIMENTS_REPORT.md` §18.
- **Limitations:** seeded single-point EDM faults on one test rig;
  small number of underlying physical experiments; licence unstated.
- **Status:** **integrated** — `CWRUAdapter`; Random Forest records
  exist (window-split, with the caveat stated in the record itself,
  and recording-wise).

## 8. Paderborn University Bearing Dataset (KAt-DataCenter)

- **Source:** Paderborn University, Chair of Design and Drive
  Technology — landing page
  https://mb.uni-paderborn.de/en/kat/research/bearing-datacenter ;
  the download page links a plain directory listing at
  https://groups.uni-paderborn.de/kat/BearingDataCenter/ (32
  per-bearing .rar archives, no form, no login, no terms
  click-through — verified live 2026-11-15; this supersedes the
  Phase-1 note that described a click-through page).
- **Licence:** **CC BY-NC 4.0 — confirmed verbatim on the official
  page** ("The data is licensed under the Creative Commons
  Attribution-NonCommercial 4.0 International Licence…
  Non-commercial academic use of the data is permitted, but a
  citation of the source is required", Lessmeier et al.). The
  Phase-1 report had this only from secondary sources; Wave 3
  resolved it. **Non-commercial only.**
- **Tasks:** A (primary — uniquely valuable because the full set
  contains *real* fatigue damage alongside artificial damage,
  enabling an artificial→real domain-shift experiment), C (secondary
  — healthy bearings as the normal class). Not B for this classic set
  (no life trajectories; a separate, much larger Paderborn
  run-to-failure dataset from 2024 exists on Zenodo and is out of
  scope here).
- **Features / labels:** synchronised motor phase currents + housing
  vibration at 64 kHz, plus speed/torque/radial-force/temperature;
  bearing type 6203; damage class per bearing (healthy / inner /
  outer / combined) and origin (artificial vs real). Integrated
  scope: **K001** (healthy), **KA01** (outer-race, artificial EDM),
  **KI01** (inner-race, artificial EDM) — archive SHA-256s recorded
  in the adapter docstring. Full set: 32 bearings (6 healthy, 12
  artificial damage, 14 real damage).
- **Size (verified from the downloaded archives):** 80 measurement
  files per bearing = 4 operating conditions (1500 rpm/0.7 Nm/1000 N;
  1500/0.1/1000; 900/0.7/1000; 1500/0.7/400) × 20 measurements × 4 s.
- **Preprocessing:** vibration_1 channel → 2,048-sample windows → the
  same 10 time-domain features as CWRU (shared feature space), 30,022
  windows total in scope. Phase currents are in the files for future
  electrical-fault work, not featurised here. Same split discipline
  as CWRU: by measurement file / bearing, never by window.
- **Status:** **integrated** (3 of 32 bearings) — `PaderbornAdapter`;
  measurement-grouped Random Forest + permutation records exist.
  **Binding caveats:** one physical bearing per class in this scope —
  no unseen-bearing generalisation claim is possible; the
  real-damage bearings are not yet downloaded.

## 9. IMS Bearing Dataset (NASA / University of Cincinnati)

- **Source:** NASA Prognostics Data Repository, entry #4 "Bearings" —
  the official page itself publishes the direct link
  https://phm-datasets.s3.amazonaws.com/NASA/4.+Bearings.zip
  (Lee, Qiu, Yu, Lin & Rexnord Technical Services, 2007). The
  Phase-1 report could not verify a bare direct link and described
  the download as small; both are corrected here — the archive is
  **1,075,597,174 bytes (≈1.08 GB)**, SHA-256
  `21001ac266c465f5d345ec42d7b508c6a6328487fd9d4d7774422dd5ea10ad83`.
- **Licence:** no per-dataset licence stated on the PCoE page
  (acknowledgement requested; use at own risk). A data.gov catalog
  entry lists a government-works licence URL, but its legacy download
  resource is not usable (redirect → 403) — the PCoE/S3 link above is
  the working official route.
- **Tasks:** C (primary), B (derived time-to-failure labels, not yet
  attempted). Not A: no per-record fault labels.
- **Features / labels:** three run-to-failure tests on one rig: 4
  Rexnord ZA-2115 double-row bearings on a common shaft, 2000 rpm,
  6000 lb radial load. Each file is a 1-second vibration snapshot
  (20,480 samples at 20 kHz) taken every 10 minutes (every 5 minutes
  for the first 43 files of Test 1). Files per test: 2,156 / 984 /
  4,448 (7,588 total). Test 1 has 2 channels per bearing (x/y);
  Tests 2–3 have 1 channel per bearing. No labels are provided;
  post-hoc documented end-of-test faults: Test 1 — bearing 3
  inner-race defect + bearing 4 rolling-element defect; Test 2 —
  bearing 1 outer-race fault; Test 3 — bearing 3 outer-race fault.
- **Preprocessing:** per-file feature extraction (RMS, kurtosis,
  crest factor…) producing one health-indicator point per snapshot;
  the adapter handles both channel layouts. RUL, if attempted, must
  be derived from file timestamps with the convention stated (the rig
  was stopped on a defect criterion, not full seizure).
- **Status:** **integrated (Test 1)** — `IMSAdapter`; Wave-5 Task-C
  record `ims__taskC__isolation_forest__s42`. Downloaded 2026-11-15
  after the user approved lifting Wave 3's size gate for this
  archive (Wave 3 had deliberately not fetched it). The archive is
  triple-nested (zip → IMS.7z → per-test .rar). Only 1st_test.rar was
  extracted: 2,156 snapshot files (2003-10-22 → 2003-11-25). Tests
  2–3 remain unextracted in the local archive. **Wave-5 outcome:**
  the Isolation Forest raised its first sustained alarm 337
  snapshots before the end of the run on bearing 3 and 723 snapshots
  before the end on bearing 4 — the two bearings with documented
  teardown defects — and never sustained an alarm on bearings 1–2.
  With no per-snapshot labels, the record is framed strictly as
  alarm timing; no precision/recall is claimed.
- **Limitations:** only 3 rig runs; bearings on a shared shaft
  cross-talk, so channels are not independent systems; no
  ground-truth RUL file — evaluation conventions vary across papers
  and must be stated.

## 10. NASA Li-ion Battery Aging

- **Source:** NASA Ames Prognostics Data Repository, dataset #5
  "Batteries" (Saha & Goebel, 2007) —
  https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/
  Downloaded 2026-11-15 from the zip link the repository page itself
  publishes (PHM Society S3 mirror):
  https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip
  (209,708,670 bytes; SHA-256
  82302a7db4fc1b34e0b6676326610438d43b816bdf11a69d1d012a464ef2f92e).
- **Licence:** no named licence is stated on the repository page or
  in the companion data.gov catalog entry; the page *requests*
  acknowledgement of the repository and the data donators in
  publications and states users employ the data at their own risk.
- **Tasks:** B (primary — capacity-fade RUL in cycles).
- **Features / labels:** per battery, a sequence of charge / discharge
  / impedance operations. Integrated scope: the 4 room-temperature
  run-to-failure batteries B0005, B0006, B0007, B0018 (inner archive
  #1; the full bundle holds 34 unique batteries). One adapter row per
  discharge cycle: capacity (Ah), SOH vs the 2 Ah rating, duration,
  voltage/temperature/current statistics (636 rows total). Labels:
  `rul_cycles_eol` — discharge cycles until capacity first falls
  below the dataset's own EOL criterion, 70% of rated = 1.4 Ah
  (crossings: B0005 @ discharge 125, B0006 @ 109, B0018 @ 97; **B0007
  never crosses** — last capacity 1.432 Ah — so its EOL label falls
  back to cycles-to-end, documented in the adapter);
  `rul_cycles_to_end` is also provided.
- **Preprocessing:** discharge cycles only (charge/impedance
  operations carry no Capacity); splits must be by battery, never by
  cycle — with 4 batteries, leave-one-battery-out is the honest
  protocol (and is what the wave-4 records use).
- **Status:** **integrated** (4 batteries) — `NASABatteryAdapter`;
  leave-one-battery-out RF and LSTM records + a SHAP record exist.

## 11. FEMTO-ST / PRONOSTIA Bearing Dataset (IEEE PHM 2012 Challenge)

- **Source:** FEMTO-ST Institute (France); created for the IEEE
  PHM 2012 Data Challenge (Nectoux et al., 2012). Distributed via the
  NASA PCoE repository and community mirrors — no single canonical
  direct link verified; treat download as "mirror-based, verify
  checksum/source at integration time".
- **Licence:** **not confirmed** — no licence text found on the pages
  consulted.
- **Tasks:** B (primary — real, not simulated, run-to-failure data
  with natural degradation; the strongest reality check for the
  C-MAPSS-trained approach), C (secondary). Not A (fault types are
  not labelled per bearing).
- **Features / labels:** 17 run-to-failure bearings (6 training
  full-life, 11 test truncated with ground-truth RUL); horizontal +
  vertical vibration snapshots (25.6 kHz, 2,560 samples every 10 s) +
  temperature; 3 operating conditions (1800 rpm/4000 N,
  1650 rpm/4200 N, 1500 rpm/5000 N); failure criterion vibration
  > 20 g; runs last ~1 h to ~7.8 h.
- **Preprocessing (planned):** per-snapshot feature extraction →
  per-bearing series (shares design with IMS/XJTU-SY); temperature
  channel is recorded at a lower rate and is gappy. Thousands of
  small headerless CSVs per bearing — ingestion effort is the main
  cost.
- **Limitations:** only 17 bearings (small-n prognostics — high
  evaluation variance); accelerated-life rig, so lifetimes are
  compressed; mirror provenance must be pinned at download time.
- **Status:** **researched-only**.

## 12. XJTU-SY Bearing Dataset

- **Source:** Xi'an Jiaotong University + Changxing Sumyoung
  Technology — http://biaowang.tech/xjtu-sy-bearing-datasets/
  (Wang, Lei, Li & Li, 2018/2019). Downloads via Drive / Dropbox /
  MEGA / MediaFire / Baidu mirrors only — no direct university-hosted
  file link; mirror availability can rot and must be checked at
  integration time.
- **Licence:** **not stated** on the official page.
- **Tasks:** B (primary — 15 complete trajectories is a healthier
  sample size than FEMTO-ST/IMS for bearing RUL, with a documented
  life table), C (secondary). Not A by design.
- **Features / labels:** 15 run-to-failure bearings (LDK UER204),
  3 conditions × 5 bearings — (2100 rpm, 12 kN), (2250 rpm, 11 kN),
  (2400 rpm, 10 kN). Each sampling = 32,768 points (1.28 s at
  25.6 kHz), taken every 1 minute; ≈9,216 recordings / ≈4.4 GB total
  (community packaging figure). Two vibration channels (horizontal +
  vertical). RUL derives from run duration; the failed element per
  bearing (inner race / outer race / cage / combined) is documented
  post-hoc in the life table. Lifetimes range ≈52 min to ≈8.9 h.
- **Preprocessing (planned):** per-minute snapshot features →
  per-bearing series (adapter can share code with FEMTO-ST);
  per-bearing normalisation choices matter given the wide lifetime
  spread.
- **Limitations:** mirror-only distribution (provenance risk);
  accelerated degradation; licence unstated.
- **Status:** **researched-only**.

## 13. Milling Tool Wear Dataset (UC Berkeley BEST Lab / NASA)

- **Source:** NASA Open Data Portal —
  https://data.nasa.gov/dataset/milling-wear ; also NASA PCoE
  repository. UC Berkeley BEST Lab (Agogino & Goebel, 2007).
- **Licence:** distributed via NASA portals; formal licence text
  **not confirmed**.
- **Tasks:** A (wear-state classification), B (flank-wear VB
  regression), C (secondary).
- **Features / labels:** 167 milling runs over 2 materials (cast
  iron, stainless J45) × DOE conditions (depth of cut 0.75/1.5 mm,
  feed 0.25/0.5 mm/rev); per-run table + workpiece vibration, acoustic
  emission (table/spindle), spindle motor current (AC/DC); label =
  measured flank wear VB (mm) per run.
- **Preprocessing (planned):** use the stable cutting window of each
  run only (file sizes vary 5,000–500,000 rows — variable-length
  series); several runs have missing VB (unlabelled — still usable
  for Task C) and a few are flagged defective/corrupted in the
  dataset's own notes — exclude and document, never silently repair.
- **Status:** **researched-only**. (This section was the truncation
  point of the Phase-1 file; it is restored here from the registry.)

## 14. Hydraulic Systems Condition Monitoring

- **Source:** University of Duisburg-Essen (ZeMA) via UCI —
  https://archive.ics.uci.edu/dataset/447/condition%2Bmonitoring%2Bof%2Bhydraulic%2Bsystems
  (DOI 10.24432/C5KW2S).
- **Licence:** CC BY 4.0.
- **Tasks:** A (primary — multi-output: cooler, valve, internal pump
  leakage, accumulator; graded condition classes, *not* failure
  events), C (secondary). Not B (fixed 60 s cycles, no
  run-to-failure).
- **Features / labels:** 2,205 cycles × 17 physical sensors
  (pressure, flow, temperature, vibration, power) at 1/10/100 Hz,
  pre-extracted to 43,680 statistical features in the release;
  `profile.txt` holds the 5 condition targets per cycle.
- **Preprocessing (planned):** use the released feature matrices (raw
  series are not in the UCI release); targets are ordinal condition
  grades — metric choice must respect that.
- **Status:** **researched-only**.

## 15. MetroPT-3 (Lisbon Metro air-production unit)

- **Source:** Metro do Porto / Hitachi Rail via UCI —
  https://archive.ics.uci.edu/dataset/791/metropt%2B3%2Bdataset
  (DOI 10.24432/C5VW3F).
- **Licence:** CC BY 4.0.
- **Tasks:** C (primary), A (derived from maintenance reports).
- **Features / labels:** 1,516,948 rows × 15 features (7 analogue
  pressures/temperature/motor current + 8 digital valve/compressor
  states), 1 reading/minute, Jan–Aug 2020; ground truth =
  timestamped maintenance reports (documented air-leak failures in
  Apr/Jun/Jul 2020), **no per-row labels**.
- **Preprocessing (planned):** derive event windows from report
  intervals and state the derivation; 208 MB CSV — chunked loading;
  evaluation must be event-based (detection delay / event recall),
  not row accuracy.
- **Status:** **researched-only**.

## 16. Bosch Production Line Performance (Kaggle)

- **Source:** Kaggle competition (Bosch, 2016) —
  https://www.kaggle.com/c/bosch-production-line-performance
  **No direct download exists:** access requires a Kaggle account
  and acceptance of the competition rules, which also blocks
  scripted ingestion.
- **Licence:** Kaggle competition rules; redistribution restricted.
- **Tasks:** A (primary, at genuine industrial scale), C (secondary).
  Not B.
- **Features / labels:** train 1,184,687 parts (test 1,183,748);
  4,264 anonymised features (968 numeric + 2,140 categorical + 1,156
  date) across 4 production lines / ~51 stations; binary `Response`,
  failure rate ≈0.58%; ≈14.3 GB of CSVs. Extreme sparsity (parts
  traverse different station subsets); competition test labels are
  not public, so evaluation must use a train split.
- **Status:** **blocked** — Kaggle account + competition-rules
  acceptance; no direct download. Deprioritised (LOW) on access
  friction + scale + zero feature interpretability, not on data
  quality.

## 17. MIMII — Malfunctioning Industrial Machine Sound (Hitachi)

- **Source:** Zenodo record 3384388 —
  https://zenodo.org/records/3384388 (DOI 10.5281/zenodo.3384388),
  published 2019-09-20 by Hitachi, Ltd. (Purohit et al.).
- **Licence:** **CC BY-SA 4.0** (as stated on the Zenodo record; do
  not confuse with the separate MIMII DUE record, which is
  CC BY-NC-SA 4.0).
- **Tasks:** C (primary — normal vs anomalous sound), A only in
  derived binary form. Not B.
- **Features / labels:** .wav audio, 10 s segments, 8-channel mic
  array, 16 kHz/16-bit, real factory noise mixed at −6/0/+6 dB SNR;
  labels = machine type (fan/pump/slider/valve) × model ID ×
  normal/anomalous. The specific fault type is **not** labelled per
  file. Public v1.0 contains only 4 of the 7 paper model IDs
  (00, 02, 04, 06).
- **Size (verified on Zenodo):** **100.2 GB** total — 12 zips
  (3 SNR × 4 machine types), each ≈6.9–10.9 GB.
- **Status:** **researched-only, by decision** — 100 GB of audio is
  disproportionate to the platform's current stage. A future wave
  could take a single machine type at a single SNR (≈7–11 GB) if
  audio anomaly detection becomes a priority.

## 18. MFPT Bearing Fault Data (MFPT Society)

- **Source:** historically https://mfpt.org/fault-data-sets/
  (Bechhoefer, "Condition Based Maintenance Fault Database…", MFPT
  Society).
- **Blocker (exact):** **the official site is gone.** As of
  2026-11-15 every mfpt.org path returns a domain-wide 301 redirect
  to https://www.asnt.org/about/managed-affiliates/mfpt/ (ASNT now
  manages MFPT), and that page contains no dataset, no download link
  and no licence text. The historical single-zip route
  (`mfpt_fault_data_sets.zip`, ≈29 MB per secondary metadata) is
  unreachable. No mirror was substituted — GitHub/Kaggle/MathWorks
  copies exist but are not the official source.
- **Licence:** not stated on the live (ASNT) page; the historical
  page's licence could not be verified (a third-party metadata claim
  of CC BY-NC-SA 4.0 is **unconfirmed** — do not rely on it).
- **Historical contents (secondary-documented, NOT verified from the
  live official page):** 23 .mat files — baseline (3, 270 lbs,
  97,656 Hz), outer-race constant-load (3), outer-race variable-load
  (7, 25–300 lbs, 48,828 Hz), inner-race variable-load (7, 0–300
  lbs), 3 real-world examples; each file one `bearing` struct (gs
  signal, sr, load, characteristic frequencies). An adapter mirroring
  CWRU would be feasible **if** the files are legitimately obtained.
- **Possible next steps (not taken — outside the
  official-source-only scope):** contact the MFPT Data Management
  focus group (datamanagement.ai@mfpt.org, as printed on the ASNT
  page) or explicitly authorise an archived copy of the official zip.
- **Status:** **blocked** — official site retired; domain redirects
  to ASNT, no data offered.

## 19. Automotive engine health — category unfilled

- **Conclusion: no verified public sensor/time-series automotive
  engine-health dataset with a clear licence was found** in the
  2026-11-15 search. The category is recorded as unfilled rather
  than inventing an entry.
- The one licence-verified hit is **not sensor data**: "Automotive
  Faults Dataset for Diagnostic and Maintenance Systems" (Zenodo,
  DOI 10.5281/zenodo.15626055, **CC BY 4.0**, 297.7 kB) — an
  expert-knowledge base (faults → symptoms → procedures, JSON +
  flowcharts) with no engine signals or per-vehicle labels; it does
  not fit a Task A/B/C sensor slot.
- Checked and **not** admitted: the widely circulated Kaggle "engine
  health" tabular set (provenance/licence on the listing could not
  be verified) and "VED Refined" (a re-uploaded OBD-II telemetry
  derivative with no fault/health labels).
- **Status:** **no verified public sensor dataset found**.

## 20. Motor electrical fault datasets — licence-verified candidates

No single canonical dataset exists; the following were verified live
2026-11-15 with licences confirmed at the host. None downloaded yet —
all are **researched-only** candidates for a future electrical-fault
wave, in rough priority order:

1. **Dataset for Sensorless Drive Diagnosis** — UCI ML Repository
   (DOI 10.24432/C5VP5F), **CC BY 4.0**. 58,509 instances × 49
   features extracted from motor current (EMD-based statistics), 11
   intact/defective drive classes × 12 operating conditions; single
   24.4 MB file. Caveat: pre-extracted features, not raw waveforms.
   Cheapest credible entry point for current-based fault
   classification (Task A).
2. **Vibration and Current Dataset of Three-Phase PMSM with Stator
   Faults** (KAIST) — Mendeley Data, DOI 10.17632/rgn5brrgrn.5,
   **CC BY 4.0** (confirmed in Mendeley's licence field). 6.97 GB;
   inter-turn and inter-coil short circuits at 8 severities + normal;
   3-phase current @ 100 kHz + vibration @ 25.6 kHz, 120 s
   recordings; backed by a *Data in Brief* 2023 paper. Directly
   on-target for stator electrical faults.
3. **UOEMD-VAFCVS** (University of Ottawa Electric Motor Dataset) —
   Mendeley Data, DOI 10.17632/msxs4vj48g.2, **CC BY 4.0**. 3.25 GB;
   classes include stator winding fault, broken rotor bars, voltage
   unbalance/single-phasing + mechanical faults; caveat: vibration +
   acoustic + temperature channels only — **no current/voltage**, so
   it supports electrical-fault *labels* via vibration, not
   electrical *signals*.
4. **NLN-EMP** (e-motor-driven centrifugal pump) — 4TU.ResearchData,
   DOI 10.4121/2b61183e-c14f-4131-829b-cc4822c369d0, **CC0**. Stator
   winding short (2 severities) + broken rotor bar among ~20 faults;
   vibration + 3-phase current + 3-phase voltage @ 20 kHz. Total size
   not verified in this pass.
5. **Broken-rotor-bar database** (USP São Carlos) — IEEE DataPort,
   DOI 10.21227/4f7v-8c57. 3-phase voltage/current + 5 vibration
   channels, 0–4 broken bars × 8 loads. **Partially verified:** page
   is labelled Open Access but no explicit licence text was found on
   it, and DataPort downloads typically need a free IEEE login —
   verify the licence on the download page before admission.
- Also noted: NOVIC+ motor compound-fault dataset (KAIST, Zenodo,
  CC BY 4.0, ≈28 GB, pre-windowed .npy) — large; compound-fault
  composition not fully enumerated in this pass.

---

## Shortlist and recommendations

**Integrated first (Phase-1 HIGH group — all done):** C-MAPSS
FD001–FD004 (the platform's working Task B, plus the cross-subset
transfer matrix), AI4I 2020 (Task A anchor), SECOM (the
missing-data/imbalance stress test; Tasks A + C), CWRU (the standard
bearing-diagnosis benchmark, with the leakage caveat carried in every
record).

**Integrated in the advanced expansion:** NASA Li-ion Battery (a
second, real-data Task B with an honest leave-one-battery-out
protocol), Paderborn (real + artificial damage; licence confirmed
CC BY-NC 4.0), IMS Test 1 (the canonical unlabelled run-to-failure
anomaly benchmark — evaluated as alarm timing only).

**Recommended next, in order:**
1. **FEMTO-ST PRONOSTIA** — the highest-value remaining Task B set
   (real run-to-failure bearings); blocked only by ingestion effort
   and mirror pinning, not by access or licence disputes.
2. **XJTU-SY** — 15 bearing RUL trajectories; same ingestion shape
   as FEMTO-ST, so the two adapters should share code. Licence
   unstated — research use only unless clarified.
3. **Paderborn real-damage bearings** — already licensed and
   partially downloaded; enables the artificial→real domain-shift
   experiment the dataset is uniquely suited for.
4. **MetroPT-3 / Hydraulic** — CC BY 4.0, direct UCI access; cheap
   Task C / multi-output Task A additions whenever a wave wants
   breadth over depth.
5. **Motor electrical candidates** — start with the UCI Sensorless
   Drive Diagnosis set (24.4 MB, CC BY 4.0) when electrical faults
   become a priority; the KAIST PMSM set is the on-target raw-signal
   option.
6. **Milling** — the only remaining set spanning all three tasks;
   moderate ingestion care needed (variable-length runs, missing
   labels handled honestly).

**Not recommended / blocked:** Bosch (Kaggle-gated, 14.3 GB,
anonymised — revisit only if scale becomes the point), MFPT (no
legitimate official source exists today), MIMII at full size
(100.2 GB; subset only, if ever), automotive engine health (no
verified dataset exists to integrate).

## Licence summary

| Dataset | Licence (exactly as recorded in `DATASET_REGISTRY.md`) |
|---|---|
| C-MAPSS FD001–FD004 | NASA research use; formal text not confirmed |
| AI4I 2020 | CC BY 4.0 |
| SECOM | CC BY 4.0 |
| CWRU Bearing | Not stated on official page |
| Paderborn Bearing | **CC BY-NC 4.0** (confirmed on official page) — non-commercial only |
| IMS Bearing | Not stated on PCoE page (acknowledgement requested; use at own risk) |
| NASA Li-ion Battery | No named licence; acknowledgement requested |
| FEMTO-ST PRONOSTIA | Not confirmed |
| XJTU-SY Bearing | Not stated |
| Milling Tool Wear | NASA portal; formal text not confirmed |
| Hydraulic Condition Monitoring | CC BY 4.0 |
| MetroPT-3 | CC BY 4.0 |
| Bosch Production Line | Kaggle competition rules |
| MIMII Sound | CC BY-SA 4.0 |
| MFPT Bearing | Not stated (live page gone) |
| Automotive engine health | — (no dataset admitted; the one verified Zenodo knowledge base is CC BY 4.0 but is not sensor data) |
| Motor electrical candidates | CC BY 4.0 / CC0 verified per candidate (USP broken-bar: licence text not yet verified) |

## Not verified (explicit list)

- Formal licence text for C-MAPSS, FEMTO-ST and Milling — none found
  on the official pages consulted; all three are flagged, not assumed.
- CWRU licence — not stated on the official download pages.
- IMS licence — not stated on the PCoE page (a data.gov entry points
  at a government-works licence URL; its download resource is dead).
- XJTU-SY licence — not stated; distribution is mirror-only, so
  provenance must be pinned at download time.
- MFPT — everything about the live source: the official site is
  retired; historical contents above are secondary-documented only,
  and the third-party CC BY-NC-SA 4.0 claim is unconfirmed.
- USP broken-rotor-bar database licence text (IEEE DataPort page
  shows Open Access but no explicit licence text was found).
- NLN-EMP total download size.
- Any automotive engine-health sensor dataset with a verifiable
  licence — none found; the category stays unfilled.

*Companion documents: `DATASET_REGISTRY.md` (operational per-dataset
record), `ARCHITECTURE.md` (platform design), `EXPERIMENTS_REPORT.md`
(every measured result), `FAULT_DIAGNOSIS.md` (what the platform can
and cannot claim to detect).*
