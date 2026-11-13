"""PREDICAI — NASA C-MAPSS FD001 dataset loading.

Single source of truth for the 26-column schema:
    unit, cycle, setting_1..3, sensor_1..21
Files are whitespace-separated and header-less (official NASA format).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS: list[str] = (
    ["unit", "cycle", "setting_1", "setting_2", "setting_3"]
    + [f"sensor_{i}" for i in range(1, 22)]
)
SETTING_COLUMNS: list[str] = ["setting_1", "setting_2", "setting_3"]
SENSOR_COLUMNS: list[str] = [f"sensor_{i}" for i in range(1, 22)]


def load_cmapss_file(path: str | Path) -> pd.DataFrame:
    """Load one C-MAPSS train/test text file with the official schema."""
    df = pd.read_csv(path, sep=r"\s+", header=None, names=COLUMNS)
    if df.shape[1] != 26:
        raise ValueError(f"{path}: expected 26 columns, got {df.shape[1]}")
    df["unit"] = df["unit"].astype(int)
    df["cycle"] = df["cycle"].astype(int)
    return df


def load_rul_file(path: str | Path) -> np.ndarray:
    """Load RUL_FD001.txt — one true-RUL value per test engine, in unit order."""
    rul = np.loadtxt(path, dtype=int)
    return rul


def load_fd001(raw_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    raw = Path(raw_dir)
    train = load_cmapss_file(raw / "train_FD001.txt")
    test = load_cmapss_file(raw / "test_FD001.txt")
    rul = load_rul_file(raw / "RUL_FD001.txt")
    return train, test, rul
