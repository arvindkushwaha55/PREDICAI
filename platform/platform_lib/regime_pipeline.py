"""Per-regime normalisation for the 6-condition C-MAPSS subsets (Wave 10).

Wave 2 scored FD002/FD004 with ONE global z-score scaler fit on the
train split (rul_pipeline.build_rul_data). That mixes six very
different operating regimes into a single mean/std per sensor, so a
sensor reading that is ordinary in one regime can sit several global
sigmas away in another. The standard remedy in the C-MAPSS literature
is regime-wise (a.k.a. mode-wise) normalisation: cluster the operating
settings into regimes, fit one scaler per regime, and scale each row
by ITS regime's statistics.

This module is build_rul_data with ONLY the scaling step changed:

- Same RUL derivation and cap (125), same engine-wise 80/20 split
  (seed 42), same std<0.005 feature-derivation rule (23 features on
  FD002/FD004), same windowing (30) and last-window test protocol.
- Regimes: KMeans (k=6) on the TRAIN split's 3 setting columns,
  settings standardised by their train-split std first so the
  small-range setting_2 is not swamped by setting_1/setting_3. On
  both FD002 and FD004 the recovered centroids land on the six
  documented nominal operating conditions (verified in the wave-10
  runner output and stored in each experiment record). The
  alternative — unique rounded setting combinations — was checked
  first and rejected on measurement: rounding to 2 decimals yields
  11 combinations (jitter splits x.00/x.01), rounding to 1 yields 7;
  neither is the documented 6-regime structure.
- Per-regime scalers are fit on TRAIN-split rows of that regime only,
  and cover the 20 SENSOR features. The 3 setting features keep the
  single global train-split z-score from wave 2: within a regime a
  setting is (near-)constant by construction, so a per-regime z-score
  for it is undefined, and the settings' job here is regime identity.
- Val/test rows are assigned to the nearest train regime centroid
  (standardised setting space). Nothing about val/test data is fitted.
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

from predicai_ml.data import SETTING_COLUMNS  # noqa: E402
from predicai_ml.preprocessing import (  # noqa: E402
    add_test_rul,
    add_train_rul,
    build_windows,
    clip_rul,
    engine_split,
    fit_scaler,
    last_window_per_engine,
)

from .rul_pipeline import RulData, derive_features  # noqa: E402

N_REGIMES = 6


@dataclass
class RegimeModel:
    """Train-fitted operating-regime clustering over the 3 settings."""

    k: int
    setting_mean: np.ndarray        # train-split mean of the settings
    setting_std: np.ndarray         # train-split std (ddof=0)
    centroids_std: np.ndarray       # (k, 3) KMeans centres, standardised space
    centroids_raw: np.ndarray       # (k, 3) same centres, original units
    train_sizes: list[int]          # train-split rows per regime

    def assign(self, df: pd.DataFrame) -> np.ndarray:
        """Nearest-centroid regime id for each row of `df` (settings only)."""
        S = df[SETTING_COLUMNS].to_numpy(dtype=np.float64)
        Z = (S - self.setting_mean) / self.setting_std
        d2 = ((Z[:, None, :] - self.centroids_std[None, :, :]) ** 2).sum(-1)
        return d2.argmin(axis=1)


def derive_regimes(train_part: pd.DataFrame, k: int = N_REGIMES,
                   seed: int = 42) -> tuple[RegimeModel, np.ndarray]:
    """Cluster the TRAIN split's settings into k regimes (train rows only).

    Returns (model, train_labels). The KMeans fit sees train-split rows
    exclusively; val/test assignment happens later via nearest centroid.
    """
    from sklearn.cluster import KMeans

    S = train_part[SETTING_COLUMNS].to_numpy(dtype=np.float64)
    mean, std = S.mean(axis=0), S.std(axis=0, ddof=0)
    Z = (S - mean) / std
    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    labels = km.fit_predict(Z)
    model = RegimeModel(
        k=k, setting_mean=mean, setting_std=std,
        centroids_std=km.cluster_centers_,
        centroids_raw=km.cluster_centers_ * std + mean,
        train_sizes=[int((labels == r).sum()) for r in range(k)],
    )
    return model, labels


@dataclass
class PerRegimeData(RulData):
    regime_info: dict = field(default_factory=dict)


def build_rul_data_perregime(train: pd.DataFrame, test: pd.DataFrame,
                             true_rul: np.ndarray, cap: int = 125,
                             seq_len: int = 30, seed: int = 42,
                             k: int = N_REGIMES) -> PerRegimeData:
    """build_rul_data with per-regime sensor scaling (see module docstring)."""
    train = clip_rul(add_train_rul(train), cap)
    test = clip_rul(add_test_rul(test, true_rul), cap)

    train_engines, val_engines = engine_split(train["unit"].unique(), 0.2, seed)
    train_part = train[train["unit"].isin(train_engines)]
    val_part = train[train["unit"].isin(val_engines)]

    features, dropped, candidate_stds = derive_features(train_part)
    sensor_features = [f for f in features if f.startswith("sensor_")]
    setting_features = [f for f in features if f.startswith("setting_")]

    regimes, train_labels = derive_regimes(train_part, k=k, seed=seed)

    # Per-regime sensor statistics, TRAIN rows of the regime only.
    global_scaler = fit_scaler(train_part, features)
    global_std = dict(zip(features, global_scaler.scale))
    regime_mean, regime_std, zero_std_fallbacks = {}, {}, []
    S_train = train_part[sensor_features].to_numpy(dtype=np.float64)
    for r in range(k):
        rows = S_train[train_labels == r]
        mu, sd = rows.mean(axis=0), rows.std(axis=0, ddof=0)
        for j, f in enumerate(sensor_features):
            if sd[j] == 0.0:      # cannot happen on these subsets; guard anyway
                sd[j] = global_std[f]
                zero_std_fallbacks.append({"regime": r, "feature": f})
        regime_mean[r], regime_std[r] = mu, sd

    def transform(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        labels = regimes.assign(df)
        vals = df[sensor_features].to_numpy(dtype=np.float64).copy()
        for r in range(k):
            m = labels == r
            if m.any():
                vals[m] = (vals[m] - regime_mean[r]) / regime_std[r]
        out[sensor_features] = vals
        # Settings keep the wave-2 global z-score (train-split statistics).
        out[setting_features] = (
            df[setting_features].to_numpy(dtype=np.float64)
            - global_scaler.mean[[features.index(f)
                                  for f in setting_features]]
        ) / global_scaler.scale[[features.index(f)
                                 for f in setting_features]]
        return out

    train_scaled = transform(train_part)
    val_scaled = transform(val_part)
    test_scaled = transform(test)

    X_train, y_train, _ = build_windows(train_scaled, features, seq_len)
    X_val, y_val, _ = build_windows(val_scaled, features, seq_len)
    X_test, y_test, test_engines = last_window_per_engine(
        test_scaled, features, seq_len)

    regime_info = {
        "method": "KMeans (k=6, seed 42, n_init=10) on the train split's "
                  "3 setting columns, standardised by train-split std; "
                  "val/test rows assigned to the nearest train centroid",
        "k": k,
        "train_sizes": regimes.train_sizes,
        "centroids_raw_settings": [
            {c: round(float(v), 4)
             for c, v in zip(SETTING_COLUMNS, regimes.centroids_raw[r])}
            for r in range(k)],
        "val_sizes": [int((regimes.assign(val_part) == r).sum())
                      for r in range(k)],
        "test_sizes": [int((regimes.assign(test) == r).sum())
                       for r in range(k)],
        "scaled_per_regime": sensor_features,
        "scaled_globally": setting_features,
        "zero_std_fallbacks": zero_std_fallbacks,
    }
    return PerRegimeData(
        X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val,
        X_test_last=X_test, y_test_last=y_test, test_engines=test_engines,
        features=features, dropped=dropped, candidate_stds=candidate_stds,
        scaler=global_scaler, train_engines=train_engines,
        val_engines=val_engines, regime_info=regime_info,
    )
