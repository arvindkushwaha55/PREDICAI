# PREDICAI — Dataset Card: NASA C-MAPSS FD001

**Source:** NASA Prognostics Center of Excellence — C-MAPSS Turbofan Engine
Degradation Simulation (Saxena et al., PHM 2008). Downloaded as the official
`CMAPSSData.zip` from data.nasa.gov (13 Nov 2026). Raw files are **not**
redistributed in this repo; fetch them with the download script (Phase A1
deliverable for the Windows runbook) or from the official source.

## Files used (FD001 only)

| File | Rows (measured) | Content |
|---|---|---|
| `train_FD001.txt` | 20,631 | 100 engines, run to failure |
| `test_FD001.txt` | 13,096 | 100 engines, truncated before failure |
| `RUL_FD001.txt` | 100 | True RUL at each test engine's last cycle |

Row counts match the published dataset description exactly. ✅

## Schema (26 columns, header-less, whitespace-separated)

`unit, cycle, setting_1..3, sensor_1..21` — defined once in
`ml/src/predicai_ml/data.py` (`COLUMNS`) and enforced by tests.

## Measured EDA findings (Phase A1, 13 Nov 2026 — from `run_eda.py`, not literature)

- **Missing values: 0** in both train and test.
- **Train engine life:** 128–362 cycles, mean 206.31.
- **Test observed life:** 31–303 cycles. **No engine (train or test) is
  shorter than 30 cycles** → the 30-cycle window always fits; the padding
  rule planned as Risk R-5 is **not needed for FD001** (kept documented for
  future subsets).
- **Test true RUL range:** 7–145 cycles.
- **Exactly constant sensors (std = 0, train):** `sensor_1, sensor_5,
  sensor_10, sensor_16, sensor_18, sensor_19` — 6 sensors, dropped.
- **Borderline:** `sensor_6` std = 0.00139 — not constant, but ~350× smaller
  than the next informative sensor. Literature usually drops it (→ 14
  features). **Decision deferred to Part 3** with a stated reason: after
  z-score standardization such a tiny-variance channel gets its noise
  amplified ~700×; the preprocessing phase will either drop it with this
  justification or keep it and let the Part 5 ablation judge. Recorded here
  so the choice is never silent.
- **Operational settings in FD001:** `setting_3` is exactly constant;
  `setting_1` / `setting_2` have near-zero variance (single operating
  condition, as documented for FD001). Settings are **not** model features
  for FD001; they are retained in the raw frame for provenance.
- **Candidate feature set going into Part 3:** 15 sensors
  (`sensor_2,3,4,6,7,8,9,11,12,13,14,15,17,20,21`), pending the sensor_6
  decision → final F is 14 or 15.

Full numbers: `ml/artifacts/eda_fd001.json`.

## RUL label rules (implemented in Part 3, tested there)

- Train: `RUL = max_cycle(engine) − cycle`
- Test (per cycle): `RUL = RUL_FD001[engine] + (max_observed_cycle − cycle)`
- Cap: piecewise-linear clipping, starting value 125 (ablation in Part 5).

## Tests

`ml/tests/test_data.py` — 3/3 passing (shapes/counts, no-missing,
per-engine cycle monotonicity).
