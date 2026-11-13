"""PREDICAI Phase A1 — Exploratory Data Analysis for FD001.

Measures (never assumes):
- shapes, engine counts, engine-life distribution
- missing values
- constant / near-constant sensors and settings (std on TRAIN only)
- sensor ranges

Writes a JSON summary to ml/artifacts/eda_fd001.json and prints a report.
Run:  python ml/scripts/run_eda.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from predicai_ml.data import SENSOR_COLUMNS, SETTING_COLUMNS, load_fd001  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "ml" / "data" / "raw"
OUT = ROOT / "ml" / "artifacts" / "eda_fd001.json"

CONST_STD_EPS = 1e-6      # exactly-constant threshold
NEAR_CONST_STD = 1e-3     # near-constant flag (relative scale noted in report)


def main() -> None:
    train, test, rul = load_fd001(RAW)

    train_lives = train.groupby("unit")["cycle"].max()
    test_lives = test.groupby("unit")["cycle"].max()

    sensor_std = train[SENSOR_COLUMNS].std()
    setting_std = train[SETTING_COLUMNS].std()

    constant_sensors = [c for c in SENSOR_COLUMNS if sensor_std[c] <= CONST_STD_EPS]
    near_const_sensors = [
        c for c in SENSOR_COLUMNS
        if CONST_STD_EPS < sensor_std[c] <= NEAR_CONST_STD
    ]
    kept_sensors = [c for c in SENSOR_COLUMNS if sensor_std[c] > NEAR_CONST_STD]

    summary = {
        "train_shape": list(train.shape),
        "test_shape": list(test.shape),
        "train_engines": int(train["unit"].nunique()),
        "test_engines": int(test["unit"].nunique()),
        "rul_values": int(len(rul)),
        "rul_min": int(rul.min()),
        "rul_max": int(rul.max()),
        "train_missing_total": int(train.isna().sum().sum()),
        "test_missing_total": int(test.isna().sum().sum()),
        "train_life_min": int(train_lives.min()),
        "train_life_max": int(train_lives.max()),
        "train_life_mean": round(float(train_lives.mean()), 2),
        "test_observed_life_min": int(test_lives.min()),
        "test_observed_life_max": int(test_lives.max()),
        "test_engines_shorter_than_30": int((test_lives < 30).sum()),
        "train_engines_shorter_than_30": int((train_lives < 30).sum()),
        "constant_sensors_std_le_1e-6": constant_sensors,
        "near_constant_sensors_std_le_1e-3": near_const_sensors,
        "kept_sensors": kept_sensors,
        "setting_std": {c: float(setting_std[c]) for c in SETTING_COLUMNS},
        "sensor_std": {c: round(float(sensor_std[c]), 6) for c in SENSOR_COLUMNS},
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nEDA summary written to {OUT}")


if __name__ == "__main__":
    main()
