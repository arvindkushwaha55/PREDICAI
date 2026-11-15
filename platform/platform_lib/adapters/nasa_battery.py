"""NASA Li-ion Battery Aging adapter (Task B — RUL in cycles).

Source: NASA Ames Prognostics Data Repository, dataset #5 "Batteries"
(Saha & Goebel, 2007), downloaded 2026-11-15 from the download link the
official repository page itself publishes (PHM Society S3 mirror of the
PCoE repository — the same route this project used for C-MAPSS):
    https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip
    (209,708,670 bytes, SHA-256
     82302a7db4fc1b34e0b6676326610438d43b816bdf11a69d1d012a464ef2f92e)
Licence: no named licence is stated on the repository page; the page
requests acknowledgement of the repository and the data donators in
publications and states users employ the data at their own risk.

Integrated scope: the four room-temperature run-to-failure batteries of
inner archive #1 (BatteryAgingARC-FY08Q4) — B0005, B0006, B0007, B0018.
(The bundle contains 34 unique batteries in total; the other archives
cover different temperatures/protocols and are out of this wave's scope.)

Each .mat holds one struct named after the battery, whose `cycle` array
mixes charge / discharge / impedance operations. The adapter emits ONE
ROW PER DISCHARGE CYCLE (the cycles that carry a measured Capacity):
per-cycle statistics plus two honestly-distinguished RUL labels —
  * rul_cycles_eol: discharge cycles remaining until the FIRST discharge
    whose capacity falls below the dataset's own end-of-life criterion,
    70% of the 2 Ah rating = 1.4 Ah (clipped at 0 after crossing);
  * rul_cycles_to_end: discharge cycles remaining until the last recorded
    discharge (tests ran slightly past the EOL crossing, so the two
    labels differ near end of life — neither is hidden).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

from .base import DatasetAdapter

BATTERIES = ["B0005", "B0006", "B0007", "B0018"]
RATED_CAPACITY_AH = 2.0
EOL_FRACTION = 0.70
EOL_CAPACITY_AH = RATED_CAPACITY_AH * EOL_FRACTION  # 1.4 Ah

FEATURE_COLUMNS = [
    "ambient_temperature", "capacity_ah", "soh", "duration_s",
    "voltage_min", "voltage_max", "voltage_mean",
    "temp_mean", "temp_max", "current_measured_mean",
]


def _scalar(x) -> float:
    return float(np.asarray(x, dtype=np.float64).ravel()[0])


def _unwrap(x):
    """scipy wraps (1,1) structs in object arrays — peel those layers."""
    while isinstance(x, np.ndarray) and x.dtype == object and x.size == 1:
        x = x.ravel()[0]
    return x


def _text(x) -> str:
    if isinstance(x, np.ndarray):
        x = x.ravel()[0]
    return str(x).strip().lower()


def _find_battery_file(raw_dir: Path, battery: str) -> Path:
    matches = sorted(raw_dir.rglob(f"{battery}.mat"))
    if not matches:
        raise FileNotFoundError(
            f"{battery}.mat not found under {raw_dir} — download the "
            f"official Battery Data Set zip (see adapter docstring)")
    return matches[0]


def load_discharge_cycles(path: Path, battery: str) -> list[dict]:
    """Per-discharge-cycle feature rows from one battery .mat file."""
    mat = loadmat(path, struct_as_record=False)
    if battery not in mat:
        raise ValueError(f"{path}: expected top-level variable {battery!r}, "
                         f"found {[k for k in mat if not k.startswith('__')]}")
    cycles = np.asarray(_unwrap(mat[battery]).cycle).ravel()
    rows = []
    for cyc in cycles:
        if _text(cyc.type) != "discharge":
            continue
        d = _unwrap(cyc.data)
        voltage = np.asarray(d.Voltage_measured, dtype=np.float64).ravel()
        current = np.asarray(d.Current_measured, dtype=np.float64).ravel()
        temp = np.asarray(d.Temperature_measured, dtype=np.float64).ravel()
        time = np.asarray(d.Time, dtype=np.float64).ravel()
        capacity = _scalar(d.Capacity)
        rows.append({
            "battery": battery,
            "ambient_temperature": _scalar(cyc.ambient_temperature),
            "capacity_ah": capacity,
            "soh": capacity / RATED_CAPACITY_AH,
            "duration_s": float(time.max()) if len(time) else 0.0,
            "voltage_min": float(voltage.min()),
            "voltage_max": float(voltage.max()),
            "voltage_mean": float(voltage.mean()),
            "temp_mean": float(temp.mean()),
            "temp_max": float(temp.max()),
            "current_measured_mean": float(current.mean()),
        })
    for i, row in enumerate(rows, start=1):
        row["cycle_index"] = i  # 1-based among discharge cycles
    return rows


class NASABatteryAdapter(DatasetAdapter):
    name = "nasa_battery"

    def load(self) -> tuple[pd.DataFrame, dict]:
        frames, per_battery = [], []
        for battery in BATTERIES:
            path = _find_battery_file(self.raw_dir, battery)
            rows = load_discharge_cycles(path, battery)
            if not rows:
                raise ValueError(f"{path}: no discharge cycles found")
            df_b = pd.DataFrame(rows)
            caps = df_b["capacity_ah"].to_numpy()
            below = np.nonzero(caps < EOL_CAPACITY_AH)[0]
            eol_idx = int(below[0]) + 1 if len(below) else None  # 1-based
            total = len(df_b)
            df_b["n_discharge_cycles"] = total
            if eol_idx is not None:
                df_b["rul_cycles_eol"] = np.clip(
                    eol_idx - df_b["cycle_index"], 0, None)
            else:
                df_b["rul_cycles_eol"] = total - df_b["cycle_index"]
            df_b["rul_cycles_to_end"] = total - df_b["cycle_index"]
            frames.append(df_b)
            per_battery.append({
                "battery": battery,
                "file": str(path.relative_to(self.raw_dir)),
                "n_discharge_cycles": int(total),
                "first_capacity_ah": float(caps[0]),
                "last_capacity_ah": float(caps[-1]),
                "eol_cycle_index": eol_idx,
            })
        df = pd.concat(frames, ignore_index=True)
        metadata = {
            "dataset": "nasa_battery",
            "source": "NASA Ames Prognostics Data Repository, dataset #5 "
                      "Batteries (Saha & Goebel 2007); zip linked from "
                      "https://www.nasa.gov/intelligent-systems-division/"
                      "discovery-and-systems-health/pcoe/"
                      "pcoe-data-set-repository/ (PHM Society S3 mirror)",
            "tasks": ["B"],
            "target": "rul_cycles_eol (discharge cycles until capacity "
                      "first falls below 1.4 Ah = 70% of the 2 Ah rating); "
                      "rul_cycles_to_end also provided",
            "maintenance_history": "absent",
            "data_nature": "real laboratory battery-aging data (charge / "
                           "discharge / impedance cycling rigs), not "
                           "field data",
            "feature_columns": FEATURE_COLUMNS,
            "rated_capacity_ah": RATED_CAPACITY_AH,
            "eol_capacity_ah": EOL_CAPACITY_AH,
            "rul_eol_fallback": "if a battery's capacity never falls "
                                "below 1.4 Ah (B0007: last = 1.432 Ah), "
                                "rul_cycles_eol falls back to "
                                "rul_cycles_to_end and eol_cycle_index "
                                "is null in the per-battery summary",
            "batteries": per_battery,
            "n_rows": int(len(df)),
        }
        return df, metadata
