"""C-MAPSS RUL preprocessing for platform subsets (Task B).

Mirrors the FD001 pipeline's discipline but re-derives the feature set
from the SUBSET'S OWN training split (ARCHITECTURE.md §3) instead of
inheriting FD001's drop list:

- Candidates: all 3 operating settings + all 21 sensors.
- Rule: drop a candidate whose population std on the engine-wise TRAIN
  split is below STD_THRESHOLD. Rationale, from the FD001 precedent:
  z-scoring a column with std s amplifies its noise by 1/s; FD001 dropped
  sensor_6 (std 0.00139, ~700x amplification). Threshold 0.005 corresponds
  to ~200x amplification. Every candidate's std is returned so the
  decision is auditable in the experiment record.
- FD003 finding this produces (measured 2026-11-13): all three settings
  fall below threshold (FD003 is a SINGLE-condition subset — the settings
  are constant setpoints; setting_3 is exactly constant), as do sensors
  1, 5, 10, 16, 18, 19. Sensor_6 is KEPT for FD003 (std ~0.018 — unlike
  FD001, it carries signal here, consistent with FD003's extra fan fault
  mode). On FD002/FD004 the same rule keeps the settings automatically,
  because there they genuinely vary across the 6 regimes.

Everything else reuses the existing tested helpers from predicai_ml:
engine-wise split, train-only z-score scaler, per-engine sliding windows,
official last-window test protocol.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

_ML_SRC = Path(__file__).resolve().parents[2] / "ml" / "src"
if str(_ML_SRC) not in sys.path:
    sys.path.insert(0, str(_ML_SRC))

from predicai_ml.data import SENSOR_COLUMNS, SETTING_COLUMNS  # noqa: E402
from predicai_ml.preprocessing import (  # noqa: E402
    Scaler,
    add_test_rul,
    add_train_rul,
    build_windows,
    clip_rul,
    engine_split,
    fit_scaler,
    last_window_per_engine,
)

STD_THRESHOLD = 0.005
CANDIDATE_COLUMNS: list[str] = SETTING_COLUMNS + SENSOR_COLUMNS


@dataclass
class RulData:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test_last: np.ndarray
    y_test_last: np.ndarray
    test_engines: np.ndarray
    features: list[str]
    dropped: list[dict]          # [{column, train_std}] for every dropped candidate
    candidate_stds: dict         # all candidates -> train-split std (audit trail)
    scaler: Scaler
    train_engines: list[int] = field(default_factory=list)
    val_engines: list[int] = field(default_factory=list)


def derive_features(train_part: pd.DataFrame,
                    threshold: float = STD_THRESHOLD) -> tuple[list[str], list[dict], dict]:
    """Data-derived feature selection on the training split only."""
    stds = train_part[CANDIDATE_COLUMNS].std(ddof=0)
    features, dropped = [], []
    for col in CANDIDATE_COLUMNS:
        s = float(stds[col])
        if s < threshold:
            dropped.append({"column": col, "train_std": s})
        else:
            features.append(col)
    return features, dropped, {c: float(stds[c]) for c in CANDIDATE_COLUMNS}


def build_rul_data(train: pd.DataFrame, test: pd.DataFrame, true_rul: np.ndarray,
                   cap: int = 125, seq_len: int = 30, seed: int = 42,
                   std_threshold: float = STD_THRESHOLD) -> RulData:
    """Full Task-B preprocessing for one C-MAPSS subset (see module docstring)."""
    train = clip_rul(add_train_rul(train), cap)
    test = clip_rul(add_test_rul(test, true_rul), cap)

    train_engines, val_engines = engine_split(train["unit"].unique(), 0.2, seed)
    train_part = train[train["unit"].isin(train_engines)]
    val_part = train[train["unit"].isin(val_engines)]

    features, dropped, candidate_stds = derive_features(train_part, std_threshold)

    scaler = fit_scaler(train_part, features)
    train_part = scaler.transform(train_part)
    val_part = scaler.transform(val_part)
    test_scaled = scaler.transform(test)

    X_train, y_train, _ = build_windows(train_part, features, seq_len)
    X_val, y_val, _ = build_windows(val_part, features, seq_len)
    X_test, y_test, test_engines = last_window_per_engine(test_scaled, features, seq_len)

    return RulData(
        X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val,
        X_test_last=X_test, y_test_last=y_test, test_engines=test_engines,
        features=features, dropped=dropped, candidate_stds=candidate_stds,
        scaler=scaler, train_engines=train_engines, val_engines=val_engines,
    )
