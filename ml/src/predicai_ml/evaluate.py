"""PREDICAI — evaluation metrics (Part 5 uses these; defined with Part 4).

All metrics are in cycles. The NASA PHM08 score is asymmetric:
over-predicting RUL (predicting more life than remains) is punished
harder than under-predicting, because it is the dangerous error.
"""
from __future__ import annotations

import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_pred) - np.asarray(y_true)) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_pred) - np.asarray(y_true))))


def nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """PHM08 scoring function (Saxena et al. 2008). Lower is better."""
    d = np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64)
    early = np.exp(-d[d < 0] / 13.0) - 1.0    # under-prediction
    late = np.exp(d[d >= 0] / 10.0) - 1.0     # over-prediction (heavier)
    return float(np.sum(early) + np.sum(late))


def evaluate_all(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {"rmse": rmse(y_true, y_pred), "mae": mae(y_true, y_pred),
            "nasa_score": nasa_score(y_true, y_pred)}
