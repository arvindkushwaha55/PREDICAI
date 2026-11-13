"""PREDICAI — preprocessing pipeline for C-MAPSS FD001 (Phase A2 / Part 3).

Locked decisions (see docs/DATASET.md and plan.md):
- Features: 14 sensors. Dropped: the 6 exactly-constant sensors
  (1, 5, 10, 16, 18, 19) and sensor_6 (std 0.00139 in train — after
  z-score standardisation its noise would be amplified ~700x; it is the
  sensor the literature standard pipeline also drops). Operational
  settings are not features in FD001 (setting_3 constant, others
  near-constant — single operating condition).
- RUL: train  RUL = max_cycle(engine) - cycle
       test   RUL = RUL_FD001[engine] + (max_observed_cycle - cycle)
- Piecewise cap: RUL <- min(RUL, cap), cap default 125 (ablation in Part 5).
- Split: engine-wise 80/20 train/val with a fixed seed. The official test
  set is never used for fitting anything.
- Scaler: z-score with population std (ddof=0, same as sklearn
  StandardScaler), fit on the TRAIN split only. The exact parameters are
  exported to preprocessing.json — the single contract the Rust engine
  will read, so Python/Rust normalization cannot drift.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES: list[str] = [
    "sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_8", "sensor_9",
    "sensor_11", "sensor_12", "sensor_13", "sensor_14", "sensor_15",
    "sensor_17", "sensor_20", "sensor_21",
]
SEQ_LEN = 30
RUL_CAP = 125
SPLIT_SEED = 42
VAL_FRACTION = 0.2


# ---------------------------------------------------------------- RUL labels
def add_train_rul(df: pd.DataFrame) -> pd.DataFrame:
    """RUL for run-to-failure training engines: max_cycle - cycle."""
    out = df.copy()
    max_cycle = out.groupby("unit")["cycle"].transform("max")
    out["rul"] = (max_cycle - out["cycle"]).astype(int)
    return out


def add_test_rul(df: pd.DataFrame, true_rul: np.ndarray) -> pd.DataFrame:
    """RUL for truncated test engines.

    true_rul[i] is the true RUL at the LAST observed cycle of engine i+1.
    At any earlier cycle t: RUL = true_rul + (max_observed_cycle - t).
    """
    out = df.copy()
    max_cycle = out.groupby("unit")["cycle"].transform("max")
    last_rul = out["unit"].map(lambda u: int(true_rul[int(u) - 1]))
    out["rul"] = (last_rul + (max_cycle - out["cycle"])).astype(int)
    return out


def clip_rul(df: pd.DataFrame, cap: int = RUL_CAP) -> pd.DataFrame:
    out = df.copy()
    out["rul"] = out["rul"].clip(upper=cap)
    return out


# --------------------------------------------------------------------- split
def engine_split(
    units: np.ndarray, val_fraction: float = VAL_FRACTION, seed: int = SPLIT_SEED
) -> tuple[list[int], list[int]]:
    """Engine-wise train/val split — engines never appear on both sides."""
    rng = np.random.default_rng(seed)
    shuffled = np.array(sorted(int(u) for u in units), dtype=int)
    rng.shuffle(shuffled)
    n_val = int(round(len(shuffled) * val_fraction))
    val = sorted(shuffled[:n_val].tolist())
    train = sorted(shuffled[n_val:].tolist())
    return train, val


# -------------------------------------------------------------------- scaler
@dataclass
class Scaler:
    features: list[str]
    mean: np.ndarray
    scale: np.ndarray

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out[self.features] = (df[self.features].to_numpy() - self.mean) / self.scale
        return out

    def to_dict(self) -> dict:
        return {
            "features": self.features,
            "mean": [float(x) for x in self.mean],
            "scale": [float(x) for x in self.scale],
        }


def fit_scaler(train_df: pd.DataFrame, features: list[str] = FEATURES) -> Scaler:
    """Fit z-score parameters on the TRAIN split only (ddof=0)."""
    values = train_df[features].to_numpy(dtype=np.float64)
    mean = values.mean(axis=0)
    scale = values.std(axis=0, ddof=0)
    if np.any(scale == 0):
        bad = [features[i] for i in np.where(scale == 0)[0]]
        raise ValueError(f"Zero-variance features cannot be scaled: {bad}")
    return Scaler(features=list(features), mean=mean, scale=scale)


# ------------------------------------------------------------------- windows
def build_windows(
    df: pd.DataFrame, features: list[str] = FEATURES, seq_len: int = SEQ_LEN
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sliding windows inside each engine (never across engines).

    Returns (X, y, window_engine):
        X: (N, seq_len, F) float32 — normalized feature sequences
        y: (N,) float32 — (clipped) RUL at the window's last cycle
        window_engine: (N,) int — engine id each window came from
    Engines shorter than seq_len contribute no windows (FD001: none —
    minimum observed life is 31, measured in EDA).
    """
    xs, ys, engines = [], [], []
    for unit, grp in df.sort_values(["unit", "cycle"]).groupby("unit"):
        values = grp[features].to_numpy(dtype=np.float32)
        labels = grp["rul"].to_numpy(dtype=np.float32)
        if len(grp) < seq_len:
            continue
        for end in range(seq_len, len(grp) + 1):
            xs.append(values[end - seq_len : end])
            ys.append(labels[end - 1])
            engines.append(int(unit))
    if not xs:
        return (
            np.empty((0, seq_len, len(features)), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=int),
        )
    return np.stack(xs), np.array(ys), np.array(engines)


def last_window_per_engine(
    df: pd.DataFrame, features: list[str] = FEATURES, seq_len: int = SEQ_LEN
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The official test protocol: exactly one window per engine — its last."""
    X, y, engines = build_windows(df, features, seq_len)
    last_idx: dict[int, int] = {}
    for i, e in enumerate(engines):
        last_idx[int(e)] = i  # windows are in cycle order; later overwrites
    idx = [last_idx[e] for e in sorted(last_idx)]
    return X[idx], y[idx], engines[idx]


# ------------------------------------------------------------------ artifact
@dataclass
class PreprocessResult:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test_last: np.ndarray
    y_test_last: np.ndarray
    test_engines: np.ndarray
    scaler: Scaler
    train_engines: list[int] = field(default_factory=list)
    val_engines: list[int] = field(default_factory=list)


def run_pipeline(
    train: pd.DataFrame, test: pd.DataFrame, true_rul: np.ndarray,
    cap: int = RUL_CAP, seq_len: int = SEQ_LEN,
) -> PreprocessResult:
    train = clip_rul(add_train_rul(train), cap)
    test = clip_rul(add_test_rul(test, true_rul), cap)

    train_engines, val_engines = engine_split(train["unit"].unique())
    train_part = train[train["unit"].isin(train_engines)]
    val_part = train[train["unit"].isin(val_engines)]

    scaler = fit_scaler(train_part)
    train_part = scaler.transform(train_part)
    val_part = scaler.transform(val_part)
    test_scaled = scaler.transform(test)

    X_train, y_train, _ = build_windows(train_part, FEATURES, seq_len)
    X_val, y_val, _ = build_windows(val_part, FEATURES, seq_len)
    X_test, y_test, test_engines = last_window_per_engine(test_scaled, FEATURES, seq_len)

    return PreprocessResult(
        X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val,
        X_test_last=X_test, y_test_last=y_test, test_engines=test_engines,
        scaler=scaler, train_engines=train_engines, val_engines=val_engines,
    )


def save_preprocessing_json(result: PreprocessResult, path: str | Path,
                            cap: int = RUL_CAP, seq_len: int = SEQ_LEN) -> None:
    """The Python<->Rust contract. Rust must read THIS file, never hardcode."""
    payload = {
        "version": 1,
        "dataset": "NASA C-MAPSS FD001",
        "features": result.scaler.features,
        "num_features": len(result.scaler.features),
        "seq_len": seq_len,
        "rul_cap": cap,
        "scaler": {"type": "zscore", "ddof": 0,
                   "mean": result.scaler.to_dict()["mean"],
                   "scale": result.scaler.to_dict()["scale"]},
        "train_engines": result.train_engines,
        "val_engines": result.val_engines,
    }
    Path(path).write_text(json.dumps(payload, indent=2))
