# PredictAI Platform Expansion — Phase 1: Dataset Research Report

**Date:** 2026-11-13
**Scope:** Multi-dataset, multi-model predictive-maintenance platform (Tasks: A = failure
classification, B = RUL regression, C = anomaly detection), expanding the existing
FD001-only PREDICAI system.
**Method:** Every factual claim below was checked against a live source (official
repository page, source-organisation page, or the dataset's own documentation) during
this research session. The source URL used is recorded per dataset. Anything that could
not be confirmed is explicitly marked **not verified** — see the final section.
**Result:** 13 dataset entries researched; 12 are distinct, usable datasets (entry 13,
the PHM 2008 Challenge data, turns out to be the same data family as C-MAPSS and is
documented as such rather than counted separately).

---

## Summary Comparison Table

| # | Dataset | A: Classification | B: RUL | C: Anomaly | Access | Priority |
|---|---------|:---:|:---:|:---:|----------|----------|
| 1 | NASA C-MAPSS FD001–FD004 | (derived) | ✅ primary | (derived) | Direct ZIP (NASA / PHM mirror) | **HIGH** |
| 2 | UCI AI4I 2020 | ✅ primary | ❌ | ✅ secondary | Direct CSV (UCI) | **HIGH** |
| 3 | CWRU Bearing | ✅ primary (fault type) | ❌ | ✅ secondary | Direct .mat files (university site) | **HIGH** |
| 4 | FEMTO-ST PRONOSTIA (IEEE PHM 2012) | ❌ | ✅ primary | ✅ secondary | NASA PCoE / GitHub mirrors | MEDIUM |
| 5 | Paderborn Bearing | ✅ primary | ❌ (classic set) | ✅ secondary | University download page (terms acceptance) | MEDIUM |
| 6 | IMS Bearing (NASA/Univ. Cincinnati) | ❌ (unlabelled) | ✅ primary | ✅ primary | NASA PCoE ZIP | MEDIUM |
| 7 | XJTU-SY Bearing | ❌ | ✅ primary | ✅ secondary | Researcher page → Drive/Dropbox/MEGA mirrors | MEDIUM |
| 8 | UCI SECOM | ✅ primary (pass/fail) | ❌ | ✅ primary | Direct files (UCI) | **HIGH** |
| 9 | Bosch Production Line (Kaggle) | ✅ primary | ❌ | ✅ secondary | Kaggle account + competition rules | LOW |
| 10 | Milling Tool Wear (UC Berkeley / NASA) | ✅ (wear-state) | ✅ (VB regression) | ✅ secondary | NASA PCoE / data.nasa.gov | MEDIUM |
| 11 | NASA Li-ion Battery Aging | ❌ | ✅ primary (capacity/RUL) | ❌ | NASA PCoE / Open Data Portal | MEDIUM |
| 12 | UCI Hydraulic Systems Condition Monitoring | ✅ primary (multi-output) | ❌ | ✅ secondary | Direct files (UCI) | MEDIUM |
| 13 | MetroPT-3 (UCI) | (derived from reports) | ❌ | ✅ primary | Direct CSV, 208 MB (UCI) | MEDIUM |
| — | PHM 2008 Challenge data | — | ✅ | — | NASA PCoE / PHM Society | LOW (same family as #1 — do not double-count) |

---

## 1. NASA C-MAPSS Turbofan Engine Degradation (FD001–FD004)

- **Name + source organisation:** Turbofan Engine Degradation Simulation Data Set;
  NASA Ames Prognostics Center of Excellence (PCoE), simulated with the C-MAPSS tool
  (Saxena & Goebel, 2008).
- **Official page URL:** NASA Prognostics Data Repository —
  https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/
  (legacy repository URL still cited by NASA-linked sources:
  https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/)
- **Direct download URL:** A PHM Society S3 mirror of the full archive is publicly
  direct-downloadable:
  https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip
- **Licence / usage terms:** US Government (NASA) work; distributed for research use
  with citation of Saxena et al. (2008). A formal licence text on the download page was
  **not verified** in this session.
- **Size (verified from published dataset tables):**

  | Subset | Train engines | Test engines | Train rows | Test rows | Operating conditions | Fault modes |
  |--------|---|---|---|---|---|---|
  | FD001 | 100 | 100 | 20,631* | 13,096* | 1 (sea level) | 1 (HPC degradation) |
  | FD002 | 260 | 259 | 53,759 | 33,991 | 6 | 1 (HPC degradation) |
  | FD003 | 100 | 100 | 24,720 | 16,596 | 1 (sea level) | 2 (HPC + fan degradation) |
  | FD004 | 249 | 248 | 61,249 | 41,214 | 6 | 2 (HPC + fan degradation) |

  *FD001 counts are from PREDICAI's own verified local copy; some published tables
  differ by ±1 row (20,632 / 13,097).
- **Features:** Per cycle: engine ID, cycle number, 3 operational settings, 21 sensor
  measurements (temperatures, pressures, speeds, ratios — continuous).
- **Target variable(s):** RUL in cycles. Training engines are run-to-failure (RUL =
  life − cycle); test engines are truncated and true RUL is given in `RUL_FD00x.txt`.
- **Suitable task(s):** **B (RUL regression)** — primary, already the platform's working
  task. A and C only in derived form (e.g. "RUL ≤ threshold" risk bands; residuals as
  anomaly scores) — there is no native failure label per cycle.
- **Format + preprocessing notes:** Space-separated TXT files. Known quirks: several
  sensors are constant in FD001 (PREDICAI drops 7 of 21); the 3 operational settings
  must be handled (or conditioned on) for the 6-condition subsets FD002/FD004, where
  raw sensor values are not comparable across conditions without normalisation per
  condition/regime; RUL is conventionally capped (piecewise-linear labels).
- **Limitations / risks:** Simulated, not real engines; single asset type; FD002/FD004
  regime shift makes naive FD001 transfer misleading — which is precisely why it is
  valuable for the cross-dataset generalisation phase (same sensors, shifted regimes
  and fault modes).
- **Integration priority: HIGH** — FD001 adapter, pipeline and Rust serving path
  already exist; FD002–FD004 reuse them and immediately enable cross-subset
  generalisation experiments at near-zero ingestion cost.

## 2. UCI AI4I 2020 Predictive Maintenance Dataset

- **Name + source organisation:** AI4I 2020 Predictive Maintenance Dataset; donated to
  the UCI Machine Learning Repository (Matzka, 2020 — synthetic dataset modelled on a
  real milling machine process).
- **Official page URL:** https://archive.ics.uci.edu/dataset/601/predictive%2Bmaintenance%2Bdata
- **Direct download URL:** Direct CSV download from the UCI page (no account or form).
  DOI: 10.24432/C5HS5C.
- **Licence / usage terms:** CC BY 4.0 (the UCI repository's standard per-dataset
  licence, shown on UCI dataset pages).
- **Size (verified on the UCI page):** 10,000 rows × 14 columns.
- **Features:** UID, product ID, product type (L/M/H quality variant), air temperature
  [K], process temperature [K], rotational speed [rpm], torque [Nm], tool wear [min].
- **Target variable(s):** Binary `Machine failure` label plus five failure-mode flags:
  TWF (tool wear), HDF (heat dissipation), PWF (power), OSF (overstrain), RNF (random).
  Failure prevalence is 339/10,000 ≈ 3.4% (verified from the UCI page's per-mode counts
  and independent analyses).
- **Suitable task(s):** **A (failure classification)** — primary: it is the cleanest
  classification-native dataset in this list, with both a binary target and per-mode
  multi-label targets. **C (anomaly detection)** — secondary (train on non-failure
  rows). Not suitable for B: rows are independent process snapshots, not
  run-to-failure trajectories, so no RUL exists.
- **Format + preprocessing notes:** Single CSV. Quirks: the failure-mode flag columns
  must be excluded from inputs when predicting `Machine failure` (target leakage);
  UID/product ID are identifiers, not features; strong class imbalance (use PR-AUC /
  recall-oriented metrics, class weighting); the underlying failure rules are
  deterministic functions of the features, so very high scores are achievable and
  should not be over-interpreted as real-world performance — the data is synthetic.
- **Limitations / risks:** Synthetic; no time/trajectory structure; failure modes are
  rule-generated. Excellent for benchmarking the classification stack and XAI
  (it was designed for explainability research), weak evidence about real machines.
- **Integration priority: HIGH** — zero-friction download, tiny size, and it is the
  anchor dataset for Task A, which the platform currently lacks entirely.

## 3. Case Western Reserve University (CWRU) Bearing Data

- **Name + source organisation:** Rolling-element bearing fault data; Bearing Data
  Center, Case Western Reserve University (Case School of Engineering).
- **Official page URL:** https://engineering.case.edu/bearingdatacenter/download-data-file
  (per-condition pages, e.g. https://engineering.case.edu/bearingdatacenter/48k-drive-end-bearing-fault-data)
- **Direct download URL:** Individual `.mat` files are directly downloadable from the
  CWRU pages (no account). Community mirrors exist but the university site is the
  canonical source.
- **Licence / usage terms:** **Not stated on the download page** (verified: the pages
  present data tables without a licence statement; academic use with citation is the
  de-facto norm — flag before any commercial use).
- **Size:** Not a single archive: one `.mat` recording per condition. Verified
  structure: fault diameters 0.007″–0.028″, motor loads 0–3 HP, drive-end data at
  12 kHz and 48 kHz sampling, fan-end at 12 kHz. Total file count / aggregate size:
  **not verified** in this session.
- **Features:** Vibration (acceleration) time series from drive-end (DE), fan-end (FE)
  and base (BA) accelerometers, plus motor RPM metadata. Bearings: SKF 6205-2RS JEM
  (drive end), SKF 6203-2RS JEM (fan end).
- **Target variable(s):** Fault class derived from the recording condition: Normal,
  Inner-Race, Ball, Outer-Race (with outer-race load-zone position @3/@6/@12 o'clock),
  crossed with fault diameter (severity) and load.
- **Suitable task(s):** **A (classification)** — primary, as multi-class fault
  *diagnosis* (note: this is diagnosis of an existing seeded fault, not prediction of
  a future failure — the dashboard wording must reflect that). **C (anomaly)** —
  secondary: train on Normal recordings only. Not suitable for B: no run-to-failure
  progression.
- **Format + preprocessing notes:** MATLAB `.mat` files (loadable via
  `scipy.io.loadmat`); each file holds `DE_time`/`FE_time`/`BA_time` arrays. Standard
  pipeline: slice signals into fixed-length windows (e.g. 1,024–4,096 samples),
  optionally FFT/statistical features. Known risks: (i) window-level random splits
  leak badly — split by recording/condition; (ii) near-ceiling accuracies (>99%) are
  common in the literature and largely reflect how separable seeded faults are;
  (iii) only one recording per condition, so "samples" are windows of a handful of
  physical runs.
- **Limitations / risks:** Seeded single-point EDM faults on one test rig; small
  number of underlying physical experiments; licence unstated.
- **Integration priority: HIGH** — direct download, the standard bearing-diagnosis
  benchmark, and its `.mat` vibration adapter is reusable for Paderborn, IMS and
  FEMTO-ST style data later.

## 4. FEMTO-ST / PRONOSTIA Bearing Dataset (IEEE PHM 2012 Challenge)

- **Name + source organisation:** PRONOSTIA bearing accelerated-degradation data;
  FEMTO-ST Institute (France), created for the IEEE PHM 2012 Data Challenge
  (Nectoux et al., 2012).
- **Official page URL:** Distributed via the NASA Prognostics Data Repository
  (https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/);
  widely used mirrors exist on GitHub, e.g.
  https://github.com/Lucky-Loek/ieee-phm-2012-data-challenge-dataset
- **Direct download URL:** No single canonical direct link verified in this session;
  access is via the NASA repository or community mirrors (GitHub/Zenodo). Treat
  download as "mirror-based, verify checksum/source at integration time".
- **Licence / usage terms:** **Not verified** — no licence text was found on the pages
  consulted in this session.
- **Size (verified):** 17 run-to-failure bearings: 6 training (full life) + 11 test
  (truncated, with ground-truth RUL at truncation). Three operating conditions:
  1800 rpm/4000 N, 1650 rpm/4200 N, 1500 rpm/5000 N. Failure criterion: vibration
  amplitude > 20 g. Runs last ~1 h to ~7.8 h.
- **Features:** Horizontal + vertical vibration (25.6 kHz; 2,560-sample snapshots
  every 10 s) and temperature (recorded at lower rate). One CSV per snapshot —
  thousands of files per bearing.
- **Target variable(s):** RUL (in seconds/minutes) at the truncation point for test
  bearings; full degradation trajectories for training bearings.
- **Suitable task(s):** **B (RUL regression)** — primary: real (not simulated)
  run-to-failure data with natural degradation, the strongest reality check for the
  C-MAPSS-trained approach. **C (anomaly/health-indicator)** — secondary.
  Not a classification dataset (fault types are not labelled per bearing).
- **Format + preprocessing notes:** Very large file counts of small headerless CSVs;
  the standard approach is per-snapshot feature extraction (RMS, kurtosis, spectral
  features) to build a per-bearing time series before any sequence modelling.
  Temperature channel has a different sampling pattern and frequent gaps.
- **Limitations / risks:** Only 17 bearings (small-n prognostics — high variance in
  evaluation); accelerated-life rig, so lifetimes are compressed; mirror provenance
  must be pinned at download time.
- **Integration priority: MEDIUM** — high scientific value for Task B realism, but
  ingestion effort (file-count, feature extraction stage) is materially larger than
  the HIGH group.

## 5. Paderborn University Bearing Dataset (KAt-DataCenter)

- **Name + source organisation:** Condition Monitoring of Bearing Damage in
  Electromechanical Drive Systems; KAt-DataCenter, Paderborn University, Germany
  (Lessmeier et al., 2016).
- **Official page URL:** https://mb.uni-paderborn.de/kat/forschung/kat-datacenter/bearing-datacenter/data-sets-and-download
  (dataset portal: https://groups.uni-paderborn.de/kat/BearingDataCenter/)
- **Direct download URL:** Per-bearing `.mat` archives from the download page;
  downloads are served after presenting the dataset terms on the site (no account
  verified as required, but it is a click-through page, not a bare file link).
- **Licence / usage terms:** Reported as **CC BY-NC 4.0** (non-commercial) by
  secondary sources; the licence text on the official download page itself was
  **not verified** in this session — confirm before integration, and note the
  non-commercial restriction would matter for any commercial use.
- **Size (verified on the official page):** 32 bearings — 6 healthy, 12 with
  artificial damage, 14 with real damage from accelerated lifetime tests.
  Four operating conditions; 20 measurements of 4 s per bearing per condition.
  (A separate, much larger Paderborn *run-to-failure* dataset from 2024 also exists
  on Zenodo; it is out of scope for the first wave and was not researched in depth.)
- **Features:** Synchronously sampled motor phase current and housing vibration,
  both at 64 kHz, plus low-rate operating parameters (speed, torque, radial force,
  temperature). Bearing type 6203.
- **Target variable(s):** Damage class per bearing: healthy / inner-race / outer-race
  / combined, and damage origin (artificial vs real), with standardised damage
  documentation sheets.
- **Suitable task(s):** **A (classification)** — primary, and uniquely valuable
  because it contains *real* fatigue damage alongside artificial damage (enables an
  artificial→real domain-shift experiment). **C (anomaly)** — secondary (healthy
  bearings as the normal class). Not B for this classic set (no life trajectories).
- **Format + preprocessing notes:** `.mat` files named by condition + bearing code
  (e.g. `N15_M07_F10_KA01_1.mat`). Current + vibration dual modality. Same
  window-leakage cautions as CWRU: split by bearing, never by window.
- **Limitations / risks:** Non-commercial licence (to confirm); 64 kHz dual-channel
  data is heavy; class taxonomy differs from CWRU (combined faults, no ball-fault
  class in the same form), so cross-dataset label alignment needs care.
- **Integration priority: MEDIUM** — strong second-wave classification dataset once
  the CWRU `.mat` adapter exists; licence confirmation is a gating task.

## 6. IMS Bearing Dataset (NASA / University of Cincinnati)

- **Name + source organisation:** Bearing Data Set; Center for Intelligent
  Maintenance Systems (IMS), University of Cincinnati, distributed via the NASA
  Ames Prognostics Data Repository (Lee et al., 2007).
- **Official page URL:** NASA Prognostics Data Repository —
  https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/
- **Direct download URL:** Repository ZIP ("IMS-Rexnord Bearing Data"); the
  repository page is the canonical route — a bare direct file link was **not
  verified** in this session.
- **Licence / usage terms:** NASA repository distribution; formal licence text
  **not verified** in this session.
- **Size (verified):** Three run-to-failure tests on one rig: 4 Rexnord ZA-2115
  double-row bearings on a common shaft, 2000 rpm, 6000 lb radial load.
  Files per test: 2,156 / 984 / 4,448 (7,588 total). Each file is a 1-second
  vibration snapshot (20,480 samples at 20 kHz), taken every 10 minutes (every
  5 minutes for the first 43 files of Test 1).
- **Features:** Accelerometer channels: Test 1 has 2 channels per bearing (x/y);
  Tests 2–3 have 1 channel per bearing (8 / 4 / 4 channels per file respectively).
- **Target variable(s):** None provided as labels. Post-hoc documented end-of-test
  faults: Test 1 — Bearing 3 inner-race defect + Bearing 4 rolling-element defect;
  Test 2 — Bearing 1 outer-race fault; Test 3 — Bearing 3 outer-race fault.
  RUL must be derived from file timestamps (time-to-end-of-test), with the caveat
  that the rig was stopped on a defect criterion, not full seizure.
- **Suitable task(s):** **C (anomaly detection)** — primary fit: long unlabelled
  mostly-healthy histories with degradation at the end; the canonical
  health-indicator/anomaly-score dataset. **B (RUL)** — primary in the literature,
  using time-to-failure as the target. Not A: no per-record fault labels.
- **Format + preprocessing notes:** Timestamped ASCII snapshot files inside a ZIP;
  standard pipeline is per-file feature extraction (RMS, kurtosis, crest factor…)
  producing one health-indicator point per 10 minutes. Channel layout differs
  between Test 1 and Tests 2–3 — the adapter must handle both.
- **Limitations / risks:** Only 3 rig runs (effectively 3–4 usable degradation
  trajectories); bearings on a shared shaft cross-talk; no ground-truth RUL file —
  evaluation conventions vary across papers, so results must state the convention
  used (e.g. which bearing, which failure time).
- **Integration priority: MEDIUM** — excellent Task C/B realism and small download,
  but derived labels and 3-run scale make it a second-wave dataset.

## 7. XJTU-SY Bearing Dataset

- **Name + source organisation:** XJTU-SY Bearing Datasets; Xi'an Jiaotong
  University (Institute of Design Science and Basic Component) with Changxing
  Sumyoung Technology Co., Ltd. (Wang, Lei, Li & Li, 2018/2019).
- **Official page URL:** http://biaowang.tech/xjtu-sy-bearing-datasets/
- **Direct download URL:** The official page links mirrors (Google Drive, Dropbox,
  MEGA, MediaFire, Baidu Netdisk) for `XJTU-SY_Bearing_Datasets.zip` — no direct
  university-hosted file link; mirror availability can rot and must be checked at
  integration time.
- **Licence / usage terms:** **Not stated / not verified** — the official page
  presents download links without licence text in what was consulted.
- **Size (verified on the official page):** 15 run-to-failure bearings (LDK UER204),
  3 operating conditions × 5 bearings: (2100 rpm, 12 kN), (2250 rpm, 11 kN),
  (2400 rpm, 10 kN). Each sampling = 32,768 points (1.28 s at 25.6 kHz), taken every
  1 minute. A community packaging reports ≈9,216 recordings / ≈4.4 GB download.
- **Features:** Two vibration channels per sampling (horizontal + vertical), CSV per
  minute of life.
- **Target variable(s):** RUL derived from run duration (each bearing's total life
  in minutes is documented); the failed element per bearing (inner race / outer
  race / cage / combined) is documented post-hoc in the dataset's life table.
- **Suitable task(s):** **B (RUL regression)** — primary: 15 complete trajectories
  is a healthier sample size than FEMTO-ST/IMS for bearing RUL, with a documented
  life table. **C (anomaly/health indicator)** — secondary. Not A by design.
- **Format + preprocessing notes:** Thousands of per-minute CSVs per bearing;
  same snapshot-feature-extraction stage as FEMTO-ST (the two adapters can share
  code). Lifetimes vary widely (≈52 min to ≈8.9 h across bearings), so per-bearing
  normalisation choices matter.
- **Limitations / risks:** Mirror-only distribution (provenance/pinning risk);
  accelerated degradation; licence unstated.
- **Integration priority: MEDIUM** — arguably the best-structured bearing RUL set
  after C-MAPSS, held back only by mirror-based access and unstated licence.

## 8. UCI SECOM Manufacturing Dataset

- **Name + source organisation:** SECOM; semiconductor manufacturing process data
  donated to the UCI Machine Learning Repository (McCann & Johnston, 2008).
- **Official page URL:** https://archive.ics.uci.edu/dataset/179/secom
- **Direct download URL:** Direct data + labels files from the UCI page (no account).
  DOI: 10.24432/C54305.
- **Licence / usage terms:** CC BY 4.0 (UCI repository standard licence).
- **Size (verified on the UCI page):** 1,567 instances × 591 features; 104 fail /
  1,463 pass (≈6.6% failure rate, ~1:14 imbalance).
- **Features:** 590 anonymised sensor/process measurements (continuous) plus a
  timestamp in the labels file. Sensor identities and units are not provided.
  (Note: UCI prose says 591 features; the released matrix has 590 measurement
  columns + label — a known off-by-one in the repository text.)
- **Target variable(s):** Binary pass (−1) / fail (+1) line-test result.
- **Suitable task(s):** **A (classification)** — primary. **C (anomaly detection)**
  — primary fit as well: only 104 labelled failures makes supervised learning
  fragile, and one-class approaches on the pass class are a standard, defensible
  treatment. Not B: no degradation trajectories.
- **Format + preprocessing notes:** Two whitespace-delimited text files (data,
  labels). Known quirks: extensive missing values (≈4.5% of cells by one verified
  analysis; some features are mostly missing), many constant/near-constant and
  duplicated features — the standard recipe is missingness filtering + imputation +
  feature selection before modelling; anonymised features mean explanations can
  only reference sensor indices, not physical quantities.
- **Limitations / risks:** Small n (1,567) against ~590 features — overfitting risk
  dominates; anonymisation blocks physical interpretation and any cross-dataset
  feature alignment; 2008 process data.
- **Integration priority: HIGH** — tiny, direct-download, and the only dataset here
  that stresses the missing-data/imbalance preprocessing path; it is the cheapest
  way to make Task C (and a second Task A) real in the first wave.

## 9. Bosch Production Line Performance (Kaggle)

- **Name + source organisation:** Bosch Production Line Performance; Bosch,
  released as a Kaggle competition (2016).
- **Official page URL:** https://www.kaggle.com/c/bosch-production-line-performance
- **Direct download URL:** **None.** Download requires a Kaggle account and
  acceptance of the competition rules.
- **Licence / usage terms:** Kaggle competition rules (data provided for the
  competition; redistribution restricted) — exact current terms **not verified**
  beyond "competition-gated".
- **Size (verified from multiple published analyses):** Train 1,184,687 parts,
  test 1,183,748 parts; 4,264 features total = 968 numeric + 2,140 categorical +
  1,156 date features across 4 production lines / ~51 stations; ≈14.3 GB of CSVs;
  failure rate ≈0.58% (6,879 failures in train).
- **Features:** Anonymised station measurements (numeric), process codes
  (categorical), and station timestamps (date), keyed by part ID.
- **Target variable(s):** Binary `Response` (part failed internal quality check).
- **Suitable task(s):** **A (classification)** — primary, at genuine industrial
  scale. **C (anomaly)** — secondary. Not B.
- **Format + preprocessing notes:** Three huge CSV families (numeric / categorical
  / date) joined on part ID; extreme sparsity (~most features missing for most
  parts, since parts traverse different station subsets); needs out-of-core or
  sampled workflows, heavy feature selection, and MCC/PR-based evaluation under
  0.58% prevalence.
- **Limitations / risks:** Kaggle-gated access blocks scripted/reproducible
  ingestion in an automated pipeline; 14 GB scale is disproportionate to a first
  wave; anonymised features prevent physical explanation; competition test labels
  are not public (evaluation must use a train split).
- **Integration priority: LOW** — real and valuable, but access friction + scale +
  zero feature interpretability make it a poor early integration; revisit once the
  benchmarking engine exists.

## 10. Milling Tool Wear Dataset (UC Berkeley BEST Lab / NASA)

- **Name + source organisation:** Mill Data Set; UC Berkeley BEST Lab (Agogino &
  Goebel, 2007), distributed via the NASA Prognostics Data Repository and the NASA
  Open Data Portal (https://data.nasa.gov/dataset/milling-wear).
- **Official page URL:** NASA Open Data Portal entry above; also listed in the NASA
  PCoE repository (see §1 URL).
- 
...[truncated 9113 chars]
---

## Addendum — 2026-11-15 (Advanced Expansion, Waves 1–3)

**File-integrity note (found this wave):** the on-disk copy of this
report ends mid-§10 (Milling) with a literal `...[truncated 9113 chars]`
marker as its last line — the file was saved truncated when Phase 1 was
written. Sections 11 (NASA Battery), 12 (Hydraulic) and 13 (MetroPT-3),
the first-wave shortlist and the "could not verify" list are therefore
absent from this file, although they were present in the Phase-1
research handoff. They are **not** re-written here (this addendum only
appends); the operational record for those datasets now lives in
`DATASET_REGISTRY.md`, which was built this wave and covers all of
them. Flagged for the parent/user to decide whether a full rewrite of
this report is wanted.

**What changed this wave (all verified live on 2026-11-15):**

1. **NASA Li-ion Battery (§11) — INTEGRATED.** The repository page's own
   download link (PHM S3 mirror of the PCoE repository) was fetched:
   `5. Battery Data Set.zip`, 209,708,670 bytes, SHA-256
   82302a7db4fc1b34e0b6676326610438d43b816bdf11a69d1d012a464ef2f92e.
   No named licence is stated (acknowledgement requested; use at own
   risk) — this supersedes §11's "NASA PCoE / Open Data Portal" access
   note. New `NASABatteryAdapter`: per-discharge-cycle table for
   B0005/B0006/B0007/B0018 (636 rows), RUL = cycles to the 1.4 Ah EOL
   criterion (B0007 never crosses — documented fallback).
2. **Paderborn (§5) — INTEGRATED (3 bearings); licence CONFIRMED.**
   The official page states **CC BY-NC 4.0** verbatim — §5's "reported
   by secondary sources, not verified on the official page" is now
   resolved. Downloads are plain per-bearing .rar archives on a public
   directory listing — no terms click-through or form (supersedes §5's
   access description). K001 (healthy), KA01 (outer-race, artificial),
   KI01 (inner-race, artificial) downloaded with SHA-256s recorded in
   `DATASET_REGISTRY.md` / the adapter; new `PaderbornAdapter`
   (vibration_1 @ 64 kHz → CWRU-shared window features; 30,022 windows).
3. **IMS (§6) — BLOCKED this wave; access facts superseded.** §6 said
   no bare direct link was verified and called the download small.
   Both are now corrected: the official PCoE page itself publishes
   https://phm-datasets.s3.amazonaws.com/NASA/4.+Bearings.zip , and it
   is **1,075,597,174 bytes (≈1.08 GB)** — over this wave's
   bounded-download gate, so it was deliberately not fetched. Licence
   remains not stated on the PCoE page.
4. **MFPT (new) — BLOCKED: official source gone.** mfpt.org now
   301-redirects domain-wide to an ASNT page with no dataset, download
   or licence. Historical contents (23 .mat files) are documented from
   secondary sources only in the registry, labelled as unverified.
   No mirror was substituted.
5. **MIMII (new) — researched-only by decision.** Zenodo record
   3384388 verified: CC BY-SA 4.0, **100.2 GB** across 12 zips —
   deliberately not downloaded.
6. **Automotive engine health (new) — no verified public sensor
   dataset found.** The only licence-verified hit (Zenodo, CC BY 4.0)
   is an expert-knowledge JSON, not sensor data; popular Kaggle engine
   sets could not be licence-verified. The registry records the
   category as unfilled rather than inventing an entry.
7. **Motor electrical faults (new) — licence-verified candidates
   found, none downloaded yet:** UCI Sensorless Drive Diagnosis
   (CC BY 4.0, current-derived features, 11 classes); KAIST PMSM
   stator-fault dataset (Mendeley, CC BY 4.0, raw current @ 100 kHz);
   UOEMD-VAFCVS (CC BY 4.0, electrical-fault labels but no current
   channel); NLN-EMP (CC0, current+voltage @ 20 kHz); USP broken-bar
   database (IEEE DataPort, licence text not yet verified). Full
   details in `DATASET_REGISTRY.md`.
8. **New central registry:** `DATASET_REGISTRY.md` now carries the
   per-dataset source / licence / tasks / features / preprocessing /
   integration-status record for all 20 dataset entries in scope,
   including the two new categories above. Statuses as of today:
   integrated = FD001–FD004, AI4I, SECOM, CWRU, NASA Battery,
   Paderborn (3 bearings); blocked = Bosch (Kaggle-gated), IMS (size
   gate), MFPT (source retired); everything else researched-only.
