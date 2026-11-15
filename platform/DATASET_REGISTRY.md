# PREDICAI Platform — Dataset Registry

**Created:** 2026-11-15 (Waves 1–3 of the advanced expansion brief)
**Companion research:** `DATASET_RESEARCH.md` (Phase 1 report, 2026-11-13, with a
dated wave addendum) — this registry is the operational summary: one row per
dataset, kept current with integration reality.
**Tasks:** A = failure/fault classification · B = RUL regression · C = anomaly detection.
**Integration status vocabulary:**
- **integrated** — adapter exists, data is local, and at least one experiment
  record exists (or, for FD001, the original PREDICAI system serves it).
- **adapter-ready** — adapter code + tests exist and data is local, but no
  experiment has been run yet.
- **researched-only** — verified in research; no local data/adapter yet.
- **blocked** — cannot be integrated as things stand; the exact reason is stated.

Licence entries quote what the official source states; "not confirmed" means
no licence text was found on the official page — it does **not** mean the data
is public domain.

---

## Registry table

| Dataset | Source | Licence | Tasks | Integration status | Adapter |
|---|---|---|---|---|---|
| C-MAPSS FD001 | NASA Ames PCoE (PHM mirror) | NASA research use; formal text not confirmed | B | **integrated** (original system + platform) | `CMapssAdapter` + `ml/` pipeline |
| C-MAPSS FD002 | NASA Ames PCoE (PHM mirror) | NASA research use; formal text not confirmed | B | **integrated** | `CMapssAdapter` |
| C-MAPSS FD003 | NASA Ames PCoE (PHM mirror) | NASA research use; formal text not confirmed | B | **integrated** | `CMapssAdapter` |
| C-MAPSS FD004 | NASA Ames PCoE (PHM mirror) | NASA research use; formal text not confirmed | B | **integrated** | `CMapssAdapter` |
| AI4I 2020 | UCI ML Repository | CC BY 4.0 | A, C | **integrated** | `AI4IAdapter` |
| SECOM | UCI ML Repository | CC BY 4.0 | A, C | **integrated** | `SECOMAdapter` |
| CWRU Bearing | Case Western Reserve Univ. | Not stated on official page | A (C secondary) | **integrated** (4 recordings) | `CWRUAdapter` |
| Paderborn Bearing | KAt-DataCenter, Paderborn Univ. | **CC BY-NC 4.0** (confirmed on official page) | A (C secondary) | **integrated** (3 of 32 bearings) | `PaderbornAdapter` |
| IMS Bearing | NASA PCoE / Univ. Cincinnati | Not stated on PCoE page | C, B | **blocked** — official zip verified but 1.08 GB, over this wave's bounded-download gate | — |
| FEMTO-ST PRONOSTIA | FEMTO-ST / NASA PCoE mirrors | Not confirmed | B (C secondary) | researched-only | — |
| XJTU-SY Bearing | Xi'an Jiaotong Univ. (researcher page) | Not stated | B (C secondary) | researched-only | — |
| MFPT Bearing | MFPT Society (mfpt.org) | Not stated (live page gone) | A | **blocked** — official site retired; domain redirects to ASNT, no data offered | — |
| MIMII Sound | Hitachi / Zenodo | CC BY-SA 4.0 | C (A derived) | researched-only (by decision — 100.2 GB) | — |
| NASA Li-ion Battery | NASA Ames PCoE | No named licence; acknowledgement requested | B | **integrated** (4 batteries) | `NASABatteryAdapter` |
| Bosch Production Line | Kaggle (Bosch) | Kaggle competition rules | A (C secondary) | **blocked** — Kaggle account + competition-rules acceptance; no direct download | — |
| Milling Tool Wear | UC Berkeley BEST / NASA | NASA portal; formal text not confirmed | A, B (C secondary) | researched-only | — |
| Hydraulic Condition Monitoring | Univ. of Duisburg-Essen / UCI | CC BY 4.0 | A (C secondary) | researched-only | — |
| MetroPT-3 | Metro do Porto / UCI | CC BY 4.0 | C (A derived) | researched-only | — |
| Automotive engine health | — | — | — | **no verified public sensor dataset found** (see §) | — |
| Motor electrical faults | UCI / Mendeley / 4TU candidates | CC BY 4.0 / CC0 (verified per candidate) | A | researched-only (licence-verified candidates, see §) | — |

---

## Integrated datasets

### C-MAPSS FD001 — NASA Turbofan Engine Degradation Simulation
- **Source / official URL:** NASA Ames Prognostics Center of Excellence,
  Prognostics Data Repository —
  https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/
  (downloaded via the PHM Society S3 mirror of the repository archive).
- **Licence:** US Government (NASA) work, distributed for research use with
  citation of Saxena et al. (2008); a formal licence text on the download
  page was not confirmed.
- **Tasks:** B (primary). A/C only in derived form (RUL risk bands, residuals).
- **Features / labels:** per cycle — unit, cycle, 3 operating settings,
  21 sensors; label = RUL in cycles (train run-to-failure; test truncated
  with provided true RUL).
- **Size (verified):** train 20,631 rows / 100 engines; test 13,096 rows /
  100 engines. 1 operating condition, 1 fault mode (HPC degradation).
- **Preprocessing:** drop constant/near-constant sensors (FD001 keeps 14 of
  21 sensors; the 3 settings are constant and dropped); z-score with
  train-engine statistics only; windows of 30 cycles; RUL capped at 125.
- **Status:** **integrated** — the original PREDICAI system (LSTM → ONNX →
  Rust API → dashboard) plus the platform `CMapssAdapter`. Maintenance
  history: absent (never invented).

### C-MAPSS FD002
- **Source / licence:** as FD001.
- **Tasks:** B.
- **Features / labels:** same 26-column schema; 6 operating conditions,
  1 fault mode.
- **Size (verified):** train 53,759 rows / 260 engines; test 33,991 rows /
  259 engines.
- **Preprocessing:** per-subset derivation keeps all 3 settings + 20 sensors
  (23 features — only sensor_16 drops); settings genuinely vary across the
  6 conditions, so they are features, not constants.
- **Status:** **integrated** — `CMapssAdapter(subset="FD002")`; Ridge + LSTM
  experiment records exist (wave 2).

### C-MAPSS FD003
- **Source / licence:** as FD001.
- **Tasks:** B.
- **Features / labels:** same schema; 1 operating condition, 2 fault modes
  (HPC + fan degradation).
- **Size (verified):** train 24,720 rows / 100 engines; test 16,596 rows /
  100 engines.
- **Preprocessing:** per-subset derivation drops all 3 settings (constant in
  this single-condition subset) + 6 constant sensors, and *keeps* sensor_6
  → 15 features (contrast FD001's 14).
- **Status:** **integrated** — `CMapssAdapter(subset="FD003")`; Ridge + LSTM
  records exist, and it is the target of the FD001→FD003 zero-shot transfer
  record.

### C-MAPSS FD004
- **Source / licence:** as FD001.
- **Tasks:** B.
- **Features / labels:** same schema; 6 operating conditions, 2 fault modes.
- **Size (verified):** train 61,249 rows / 249 engines; test 41,214 rows /
  248 engines.
- **Preprocessing:** as FD002 (23 features; only sensor_16 drops).
- **Status:** **integrated** — `CMapssAdapter(subset="FD004")`; Ridge + LSTM
  records exist (wave 2). Note: on FD004 the LSTM wins RMSE/MAE but loses
  the NASA score to Ridge — recorded as measured.

### AI4I 2020 Predictive Maintenance
- **Source / official URL:** UCI ML Repository —
  https://archive.ics.uci.edu/dataset/601/predictive%2Bmaintenance%2Bdata
  (DOI 10.24432/C5HS5C).
- **Licence:** CC BY 4.0.
- **Tasks:** A (primary), C (secondary).
- **Features / labels:** 10,000 × 14 — air/process temperature, rotational
  speed, torque, tool wear, product type; binary `Machine failure` label
  (339 failures, ≈3.4%) + 5 failure-mode flags (TWF/HDF/PWF/OSF/RNF).
- **Preprocessing:** the 5 mode flags *define* the target and are declared
  leakage columns (never features); UDI/Product ID are identifiers; strong
  imbalance → PR-oriented metrics.
- **Status:** **integrated** — `AI4IAdapter`; 5 classifier records +
  permutation importance + SHAP record exist. Data is **synthetic**
  (rule-generated failures) — every downstream claim must keep that caveat.

### SECOM Manufacturing
- **Source / official URL:** UCI ML Repository —
  https://archive.ics.uci.edu/dataset/179/secom (DOI 10.24432/C54305).
- **Licence:** CC BY 4.0.
- **Tasks:** A and C (both primary fits).
- **Features / labels:** 1,567 rows × 590 anonymised sensor columns;
  pass (−1) / fail (+1) → platform target fail = 1 (104 fails, ≈6.6%).
- **Preprocessing:** heavy missingness (≈4.5% of cells; some features
  mostly missing) → missingness filtering + imputation + constant-feature
  drop (wave-1 pipeline: 590 → 450 surviving features, fitted on train
  only). Features are anonymised — explanations may cite sensor indices
  only, never physical names.
- **Status:** **integrated** — `SECOMAdapter`; LR/RF classification records
  + Isolation Forest anomaly record exist.

### CWRU Bearing (Case Western Reserve University)
- **Source / official URL:** CWRU Bearing Data Center —
  https://engineering.case.edu/bearingdatacenter/download-data-file
  (recordings served as individual .mat files from the university host).
- **Licence:** **not stated** on the official download pages (verified in
  Phase 1 research). Academic use with citation is the de-facto norm; flag
  before any commercial use.
- **Tasks:** A (fault *diagnosis* — the wording matters: this detects an
  existing seeded fault, it does not predict a future failure). C secondary
  (train on Normal only). Not B.
- **Features / labels:** drive-end vibration at 12 kHz; classes Normal /
  InnerRace / Ball / OuterRace. Integrated scope: exactly 4 recordings,
  0 HP, 0.007″ faults — 97.mat Normal (243,938 samples), 105.mat IR007_0
  (121,265), 118.mat B007_0 (122,571), 130.mat OR007@6_0 (121,991);
  identity self-verified via the `X<id>_DE_time` variables.
- **Preprocessing:** non-overlapping 2,048-sample windows → 10 time-domain
  features per window (296 windows total). **Binding caveat:** one
  recording per class, so window-wise splits share recordings across
  train/test — scores measure within-recording separability, not
  generalisation.
- **Status:** **integrated** — `CWRUAdapter`; Random Forest record exists
  with the caveat stated in the record itself.

### NASA Li-ion Battery Aging
- **Source / official URL:** NASA Ames Prognostics Data Repository,
  dataset #5 "Batteries" (Saha & Goebel, 2007) —
  https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/
  Downloaded 2026-11-15 from the zip link the repository page itself
  publishes (PHM Society S3 mirror of the PCoE repository — the same
  route used for C-MAPSS):
  https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip
  (209,708,670 bytes; SHA-256
  82302a7db4fc1b34e0b6676326610438d43b816bdf11a69d1d012a464ef2f92e).
- **Licence:** no named licence is stated on the repository page or in
  the companion data.gov catalog entry; the page *requests*
  acknowledgement of the repository and the data donators in
  publications and states users employ the data at their own risk.
- **Tasks:** B (primary — capacity-fade RUL in cycles).
- **Features / labels:** per battery, a sequence of charge / discharge /
  impedance operations. Integrated scope: the 4 room-temperature
  run-to-failure batteries B0005, B0006, B0007, B0018 (inner archive #1;
  the full bundle holds 34 unique batteries). One adapter row per
  discharge cycle: capacity (Ah), SOH vs the 2 Ah rating, duration,
  voltage/temperature/current statistics (636 rows total).
- **Labels:** `rul_cycles_eol` — discharge cycles until capacity first
  falls below the dataset's own EOL criterion, 70% of rated = 1.4 Ah
  (crossings: B0005 @ discharge 125, B0006 @ 109, B0018 @ 97; **B0007
  never crosses** — last capacity 1.432 Ah — so its EOL label falls back
  to cycles-to-end, documented in the adapter); `rul_cycles_to_end` is
  also provided since tests ran slightly past the crossing.
- **Preprocessing:** discharge cycles only (charge/impedance operations
  carry no Capacity); splits must be by battery, never by cycle, in any
  future experiment — with 4 batteries, leave-one-battery-out is the
  honest protocol.
- **Status:** **integrated** — `NASABatteryAdapter`; no experiment run
  yet (adapter + tests only this wave).

### Paderborn Bearing (KAt-DataCenter)
- **Source / official URL:** Paderborn University, Chair of Design and
  Drive Technology — landing page
  https://mb.uni-paderborn.de/en/kat/research/bearing-datacenter ;
  download page links a plain directory listing at
  https://groups.uni-paderborn.de/kat/BearingDataCenter/ (32 per-bearing
  .rar archives, no form, no login, no terms click-through — verified
  live 2026-11-15).
- **Licence:** **CC BY-NC 4.0 — confirmed verbatim on the official
  page** ("The data is licensed under the Creative Commons
  Attribution-NonCommercial 4.0 International Licence… Non-commercial
  academic use of the data is permitted, but a citation of the source
  is required", Lessmeier et al.). **Non-commercial only.**
- **Tasks:** A (primary), C (secondary).
- **Features / labels:** synchronised motor phase currents + housing
  vibration at 64 kHz, plus speed/torque/radial-force/temperature;
  bearing type 6203; damage class per bearing (healthy / inner / outer /
  combined) and origin (artificial vs real fatigue). Integrated scope:
  **K001** (healthy), **KA01** (outer-race, artificial EDM), **KI01**
  (inner-race, artificial EDM) — archive SHA-256s recorded in the
  adapter docstring.
- **Size (verified from the downloaded archives):** 80 measurement
  files per bearing = 4 operating conditions (1500 rpm/0.7 Nm/1000 N;
  1500/0.1/1000; 900/0.7/1000; 1500/0.7/400) × 20 measurements × 4 s.
- **Preprocessing:** vibration_1 channel → 2,048-sample windows → the
  same 10 time-domain features as CWRU (shared feature space), 30,022
  windows total. Phase currents are in the files for future
  electrical-fault work, not featurised here.
- **Status:** **integrated** — `PaderbornAdapter`; no experiment run yet.
  **Binding caveats:** one physical bearing per class in this scope —
  split by measurement file at minimum, and no unseen-bearing
  generalisation claim is possible; the dataset's *real*-damage bearings
  (the artificial→real domain-shift opportunity) are not yet downloaded.

---

## Researched-only datasets (verified, not yet integrated)

### FEMTO-ST / PRONOSTIA Bearing (IEEE PHM 2012)
- **Source:** FEMTO-ST Institute; distributed via NASA PCoE repository and
  community mirrors (no single canonical direct link verified).
- **Licence:** not confirmed — no licence text found on the pages consulted.
- **Tasks:** B (primary), C (secondary).
- **Features / labels:** 17 run-to-failure bearings (6 train full-life,
  11 test truncated with ground-truth RUL); horizontal + vertical vibration
  snapshots (25.6 kHz, 2,560 samples every 10 s) + temperature; 3 operating
  conditions; failure criterion vibration > 20 g.
- **Preprocessing (planned):** per-snapshot feature extraction → per-bearing
  series (shares design with IMS/XJTU-SY); temperature channel is gappy.
- **Status:** **researched-only** — ingestion effort (thousands of small
  CSVs) and mirror provenance pinning put it behind this wave's priorities.

### XJTU-SY Bearing
- **Source:** Xi'an Jiaotong University + Changxing Sumyoung —
  http://biaowang.tech/xjtu-sy-bearing-datasets/ (downloads via Drive /
  Dropbox / MEGA / MediaFire / Baidu mirrors only).
- **Licence:** not stated on the official page.
- **Tasks:** B (primary), C (secondary).
- **Features / labels:** 15 run-to-failure bearings (LDK UER204),
  3 conditions × 5 bearings; 2 vibration channels, 32,768-point snapshots
  every minute; documented life table incl. failed element per bearing;
  ≈4.4 GB total (community packaging figure).
- **Preprocessing (planned):** per-minute snapshot features → per-bearing
  series; lifetimes range ≈52 min to ≈8.9 h.
- **Status:** **researched-only** — mirror-only distribution (provenance
  risk) and unstated licence.

### Milling Tool Wear (UC Berkeley BEST Lab)
- **Source:** NASA Open Data Portal —
  https://data.nasa.gov/dataset/milling-wear ; also NASA PCoE repository.
- **Licence:** distributed via NASA portals; formal licence text not
  confirmed.
- **Tasks:** A (wear-state), B (flank-wear VB regression), C (secondary).
- **Features / labels:** 167 milling runs over 2 materials (cast iron,
  stainless J45) × DOE conditions (DOC 0.75/1.5 mm, feed 0.25/0.5 mm/rev);
  per-run table + workpiece vibration, AE (table/spindle), spindle motor
  current (AC/DC); label = measured flank wear VB (mm) per run.
- **Preprocessing (planned):** use the stable cutting window of each run
  only (file sizes vary 5,000–500,000 rows — variable-length series);
  several runs have missing VB (unlabelled — still usable for C) and a few
  are flagged defective/corrupted in the dataset's own notes — exclude and
  document, never silently repair.
- **Status:** **researched-only**.

### Hydraulic Systems Condition Monitoring
- **Source:** University of Duisburg-Essen (ZeMA) via UCI —
  https://archive.ics.uci.edu/dataset/447/condition%2Bmonitoring%2Bof%2Bhydraulic%2Bsystems
  (DOI 10.24432/C5KW2S).
- **Licence:** CC BY 4.0.
- **Tasks:** A (multi-output: cooler, valve, internal pump leakage,
  accumulator — graded condition classes, *not* failure events), C
  (secondary). Not B (fixed 60 s cycles, no run-to-failure).
- **Features / labels:** 2,205 cycles × 17 physical sensors (pressure,
  flow, temperature, vibration, power) at 1/10/100 Hz, pre-extracted to
  43,680 statistical features in the release; `profile.txt` holds the
  5 condition targets per cycle.
- **Preprocessing (planned):** use the released feature matrices (raw
  series are not in the UCI release); targets are ordinal condition
  grades — metric choice must respect that.
- **Status:** **researched-only**.

### MetroPT-3 (Lisbon Metro air-production unit)
- **Source:** Metro do Porto / Hitachi Rail via UCI —
  https://archive.ics.uci.edu/dataset/791/metropt%2B3%2Bdataset
  (DOI 10.24432/C5VW3F).
- **Licence:** CC BY 4.0.
- **Tasks:** C (primary), A (derived from maintenance reports).
- **Features / labels:** 1,516,948 rows × 15 features (7 analogue
  pressures/temperature/motor current + 8 digital valve/compressor
  states), 1 reading/minute, Jan–Aug 2020; ground truth = timestamped
  maintenance reports (documented air-leak failures in Apr/Jun/Jul 2020),
  **no per-row labels**.
- **Preprocessing (planned):** derive event windows from report intervals
  and state the derivation; 208 MB CSV — chunked loading; evaluation must
  be event-based (detection delay / event recall), not row accuracy.
- **Status:** **researched-only**.

### Bosch Production Line Performance
- **Source:** Kaggle competition (Bosch, 2016) —
  https://www.kaggle.com/c/bosch-production-line-performance
- **Licence:** Kaggle competition rules; redistribution restricted.
- **Tasks:** A (primary), C (secondary).
- **Features / labels:** 1,184,687 train parts; 4,264 anonymised features
  (968 numeric + 2,140 categorical + 1,156 date) across 4 lines / ~51
  stations; binary `Response`, failure rate ≈0.58%; ≈14.3 GB.
- **Status:** **blocked** — no direct download exists; access requires a
  Kaggle account and acceptance of competition rules, which also blocks
  scripted ingestion. Deprioritised (LOW) on access friction + scale +
  zero feature interpretability, not on data quality.

### MIMII — Malfunctioning Industrial Machine Sound (Hitachi)
- **Source / official URL:** Zenodo record 3384388 —
  https://zenodo.org/records/3384388 (DOI 10.5281/zenodo.3384388),
  published 2019-09-20 by Hitachi, Ltd. (Purohit et al.).
- **Licence:** **CC BY-SA 4.0** (as stated on the Zenodo record; do not
  confuse with the separate MIMII DUE record, which is CC BY-NC-SA 4.0).
- **Tasks:** C (primary — normal vs anomalous sound), A only in derived
  binary form. Not B.
- **Features / labels:** .wav audio, 10 s segments, 8-channel mic array,
  16 kHz/16-bit, real factory noise mixed at −6/0/+6 dB SNR; labels =
  machine type (fan/pump/slider/valve) × model ID × normal/anomalous.
  The specific fault type is **not** labelled per file. Public v1.0
  contains only 4 of the 7 paper model IDs (00, 02, 04, 06).
- **Size (verified on Zenodo):** **100.2 GB** total — 12 zips
  (3 SNR × 4 machine types), each ≈6.9–10.9 GB.
- **Status:** **researched-only, by decision** — not downloaded this
  wave (or planned at full size): 100 GB of audio is disproportionate
  to the platform's current stage. A future wave could take a single
  machine type at a single SNR (≈7–11 GB) if audio anomaly detection
  becomes a priority.

### Motor electrical fault datasets — licence-verified candidates
No single canonical dataset exists; the following were verified live
2026-11-15 with licences confirmed at the host. None downloaded yet —
all are **researched-only** candidates for a future electrical-fault
wave, in rough priority order:
1. **Dataset for Sensorless Drive Diagnosis** — UCI ML Repository
   (DOI 10.24432/C5VP5F), **CC BY 4.0**. 58,509 instances × 49 features
   extracted from motor current (EMD-based statistics), 11 intact/
   defective drive classes × 12 operating conditions; single 24.4 MB
   file. Caveat: pre-extracted features, not raw waveforms. Cheapest
   credible entry point for current-based fault classification (Task A).
2. **Vibration and Current Dataset of Three-Phase PMSM with Stator
   Faults** (KAIST) — Mendeley Data, DOI 10.17632/rgn5brrgrn.5,
   **CC BY 4.0** (confirmed in Mendeley's licence field). 6.97 GB;
   inter-turn and inter-coil short circuits at 8 severities + normal;
   3-phase current @ 100 kHz + vibration @ 25.6 kHz, 120 s recordings;
   backed by a *Data in Brief* 2023 paper. Directly on-target for
   stator electrical faults.
3. **UOEMD-VAFCVS** (University of Ottawa Electric Motor Dataset) —
   Mendeley Data, DOI 10.17632/msxs4vj48g.2, **CC BY 4.0**. 3.25 GB;
   classes include stator winding fault, broken rotor bars, voltage
   unbalance/single-phasing + mechanical faults; caveat: vibration +
   acoustic + temperature channels only — **no current/voltage**, so it
   supports electrical-fault *labels* via vibration, not electrical
   *signals*.
4. **NLN-EMP** (e-motor-driven centrifugal pump) — 4TU.ResearchData,
   DOI 10.4121/2b61183e-c14f-4131-829b-cc4822c369d0, **CC0**. Stator
   winding short (2 severities) + broken rotor bar among ~20 faults;
   vibration + 3-phase current + 3-phase voltage @ 20 kHz. Total size
   not verified in this pass.
5. **Broken-rotor-bar database** (USP São Carlos) — IEEE DataPort,
   DOI 10.21227/4f7v-8c57. 3-phase voltage/current + 5 vibration
   channels, 0–4 broken bars × 8 loads. **Partially verified:** page is
   labelled Open Access but no explicit licence text was found on it,
   and DataPort downloads typically need a free IEEE login — verify the
   licence on the download page before admission.
- Also noted: NOVIC+ motor compound-fault dataset (KAIST, Zenodo,
  CC BY 4.0, ≈28 GB, pre-windowed .npy) — large; compound-fault
  composition not fully enumerated in this pass.

### Automotive engine health
- **Conclusion: no verified public sensor/time-series automotive
  engine-health dataset with a clear licence was found** in this wave's
  search (2026-11-15). The registry therefore records the category as
  unfilled rather than inventing an entry.
- The one licence-verified hit is **not sensor data**: "Automotive
  Faults Dataset for Diagnostic and Maintenance Systems" (Zenodo,
  DOI 10.5281/zenodo.15626055, **CC BY 4.0**, 297.7 kB) — an
  expert-knowledge base (faults → symptoms → procedures, JSON +
  flowcharts) with no engine signals or per-vehicle labels; it does not
  fit a Task A/B/C sensor slot.
- Checked and **not** admitted: the widely circulated Kaggle "engine
  health" tabular set (provenance/licence on the listing could not be
  verified) and "VED Refined" (a re-uploaded OBD-II telemetry
  derivative with no fault/health labels).

---

## Blocked datasets (verified route exists or existed — cannot integrate now)

### IMS Bearing (NASA / University of Cincinnati)
- **Source / official URL:** NASA Prognostics Data Repository, entry
  #4 "Bearings" — the official page itself publishes the direct link
  https://phm-datasets.s3.amazonaws.com/NASA/4.+Bearings.zip
  (citation: Lee, Qiu, Yu, Lin & Rexnord Technical Services, 2007).
- **Licence:** no per-dataset licence stated on the PCoE page
  (acknowledgement requested; use at own risk). A data.gov catalog
  entry lists a government-works licence URL, but its legacy download
  resource is not usable (redirect → 403) — the PCoE/S3 link above is
  the working official route.
- **Tasks:** C (primary), B (derived time-to-failure labels).
- **Blocker (exact):** the official zip is **1,075,597,174 bytes
  (≈1.08 GB)** — verified by HEAD request 2026-11-15 — which exceeds
  this wave's bounded-download gate (~800 MB), so it was deliberately
  **not** downloaded. (Phase 1 research called IMS a "small download";
  that is superseded — it is not small.) Next step: a future wave with
  the gate raised for this one archive can fetch it and extract Test 1
  only (2,156 snapshot files, 8 channels/file). A secondary, unverified
  community note claims the archive is nested (zip → 7z → rar) —
  extraction tooling may need checking at that time.

### MFPT Bearing Fault Data (MFPT Society)
- **Source:** historically https://mfpt.org/fault-data-sets/ (Bechhoefer,
  "Condition Based Maintenance Fault Database…", MFPT Society).
- **Blocker (exact):** **the official site is gone.** As of 2026-11-15
  every mfpt.org path returns a domain-wide 301 redirect to
  https://www.asnt.org/about/managed-affiliates/mfpt/ (ASNT now manages
  MFPT), and that page contains no dataset, no download link, and no
  licence text. The historical single-zip route
  (`mfpt_fault_data_sets.zip`, ≈29 MB per secondary metadata) is
  unreachable. No mirror was substituted — GitHub/Kaggle/MathWorks
  copies exist but are not the official source.
- **Licence:** not stated on the live (ASNT) page; the historical
  page's licence could not be verified (a third-party metadata claim
  of CC BY-NC-SA 4.0 is **unconfirmed** — do not rely on it).
- **Historical contents (secondary-documented, NOT verified from the
  live official page):** 23 .mat files — baseline (3, 270 lbs,
  97,656 Hz), outer-race constant-load (3), outer-race variable-load
  (7, 25–300 lbs, 48,828 Hz), inner-race variable-load (7, 0–300 lbs),
  3 real-world examples; each file one `bearing` struct (gs signal, sr,
  load, characteristic frequencies). An adapter mirroring CWRU would be
  feasible **if** the files are legitimately obtained.
- **Possible next steps (not taken — outside this wave's
  official-source-only scope):** contact the MFPT Data Management
  focus group (datamanagement.ai@mfpt.org, as printed on the ASNT
  page) or explicitly authorise an archived copy of the official zip.

---
