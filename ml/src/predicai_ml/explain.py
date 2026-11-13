"""PREDICAI — explainability (Part 6 / Phase A5).

Two primary methods, per docs/RESEARCH_REPORT.md section 9. SHAP is
secondary / experimental on raw LSTM sequences (it breaks temporal
dependence and gives unstable rankings on C-MAPSS), so it is deliberately
NOT used here.

1. Permutation importance (model-agnostic): baseline RMSE on the
   validation windows, then one sensor at a time is shuffled across
   windows — the RMSE increase measures how much the model's accuracy
   relies on that sensor's alignment with each window.
2. Integrated Gradients (implemented manually — no Captum dependency):
   attributions along the straight-line path from a zero baseline to
   each window, in NORMALIZED space. Zero there is the train-mean
   operating point (the scaler's origin), so attributions answer:
   "how does this window's deviation from an average engine's readings
   move the prediction away from the average-engine prediction?"
   Attributions are in cycles: by the completeness property they sum
   (up to numerical error) to f(window) - f(baseline window).

Language rule (plan): results describe ASSOCIATION — which sensors the
model relies on — never causation ("sensor X causes failure").
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from predicai_ml.evaluate import rmse

PERMUTATION_SEED = 123
IG_SEED = 123
IG_STEPS = 50
IG_N_WINDOWS = 200


# ------------------------------------------------------------------ helpers
def predict_numpy(
    model: nn.Module, x: np.ndarray, batch_size: int = 1024
) -> np.ndarray:
    """Inference-convention predictions (eval mode, no grad, clipped >= 0).

    Same convention as model.predict_rul, batched over numpy input.
    """
    model.eval()
    preds: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            chunk = torch.from_numpy(np.asarray(x[start : start + batch_size]))
            preds.append(torch.clamp(model(chunk), min=0.0).numpy())
    return np.concatenate(preds)


# ------------------------------------------------------- permutation (1)
def permutation_importance(
    model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    features: list[str],
    seed: int = PERMUTATION_SEED,
    n_repeats: int = 1,
    batch_size: int = 1024,
) -> dict:
    """Per-sensor permutation importance on a window set.

    For each sensor, that sensor's whole 30-cycle trace is shuffled
    across windows (a different window's trace replaces it), which
    destroys the sensor's per-window information while keeping its
    marginal distribution. Returns a dict with the baseline RMSE and a
    ranking sorted by RMSE increase (descending). Deterministic for a
    fixed seed: one RNG stream, one permutation per (sensor, repeat).
    """
    x = np.asarray(x)
    y = np.asarray(y)
    baseline = rmse(y, predict_numpy(model, x, batch_size))

    rng = np.random.default_rng(seed)
    ranking: list[dict] = []
    for j, name in enumerate(features):
        permuted_rmses: list[float] = []
        for _ in range(n_repeats):
            order = rng.permutation(len(x))
            shuffled = x.copy()
            shuffled[:, :, j] = x[order, :, j]
            permuted_rmses.append(rmse(y, predict_numpy(model, shuffled, batch_size)))
        rmse_permuted = float(np.mean(permuted_rmses))
        ranking.append(
            {
                "sensor": name,
                "rmse_permuted": rmse_permuted,
                "rmse_increase": rmse_permuted - baseline,
            }
        )
    ranking.sort(key=lambda row: row["rmse_increase"], reverse=True)
    return {
        "rmse_baseline": baseline,
        "ranking": ranking,
        "params": {
            "seed": seed,
            "n_repeats": n_repeats,
            "n_windows": int(len(x)),
            "shuffle_unit": "whole sensor trace across windows",
        },
    }


# ------------------------------------------------- integrated gradients (2)
@dataclass
class IGResult:
    """Attributions for a set of windows plus completeness diagnostics."""

    attributions: np.ndarray  # (N, T, F) float32, cycles of predicted RUL
    predictions: np.ndarray  # (N,) raw model outputs f(window)
    baseline_prediction: float  # raw model output f(zero window)
    completeness_mean_abs_error: float  # cycles
    completeness_mean_rel_error: float  # |err| / mean|f(x) - f(baseline)|


def integrated_gradients(
    model: nn.Module,
    x: np.ndarray,
    steps: int = IG_STEPS,
    batch_size: int = 50,
) -> IGResult:
    """Integrated Gradients from the zero baseline, manual implementation.

    Path: baseline + alpha * (window - baseline), alpha in (0, 1],
    sampled at `steps` midpoints (a midpoint Riemann sum, error
    O(1/steps^2)). Gradients are of the RAW model output — the >= 0 clip
    is a post-processing step, and completeness only holds for the
    function actually differentiated. Model must be in eval mode so
    dropout cannot corrupt the gradients; this function sets it.
    """
    model.eval()
    windows = torch.from_numpy(np.asarray(x, dtype=np.float32))
    baseline = torch.zeros_like(windows)
    n, _, _ = windows.shape

    alphas = (torch.arange(steps, dtype=torch.float32) + 0.5) / steps
    attributions = torch.zeros_like(windows)
    predictions = torch.zeros(n, dtype=torch.float32)

    with torch.no_grad():
        baseline_prediction = float(model(baseline[:1]).item())

    for start in range(0, n, batch_size):
        chunk = windows[start : start + batch_size]  # (B, T, F)
        base_chunk = baseline[start : start + batch_size]
        diff = chunk - base_chunk  # (B, T, F)
        # (steps*B, T, F): every interpolation point of every window.
        points = (
            base_chunk.unsqueeze(0)
            + alphas.view(-1, 1, 1, 1) * diff.unsqueeze(0)
        ).reshape(-1, *chunk.shape[1:])
        points.requires_grad_(True)
        grads = torch.autograd.grad(model(points).sum(), points)[0]
        grads = grads.reshape(steps, *chunk.shape).mean(dim=0)
        attributions[start : start + batch_size] = diff * grads
        with torch.no_grad():
            predictions[start : start + batch_size] = model(chunk)

    attr_sum = attributions.sum(dim=(1, 2)).numpy()
    delta = predictions.numpy() - baseline_prediction
    abs_error = np.abs(attr_sum - delta)
    mean_abs_delta = float(np.mean(np.abs(delta)))
    return IGResult(
        attributions=attributions.numpy(),
        predictions=predictions.numpy(),
        baseline_prediction=baseline_prediction,
        completeness_mean_abs_error=float(np.mean(abs_error)),
        completeness_mean_rel_error=float(np.mean(abs_error) / mean_abs_delta)
        if mean_abs_delta > 0
        else float("nan"),
    )


def summarize_ig(attributions: np.ndarray, features: list[str]) -> dict:
    """Aggregate (N, T, F) attributions into per-sensor / per-timestep tables.

    Per-sensor ranking uses mean |attribution| over windows and timesteps
    (how strongly the model leans on the sensor, regardless of sign);
    the signed mean is kept alongside (direction the sensor's deviations
    push the prediction, on average). Association only — see module doc.
    """
    attrs = np.asarray(attributions, dtype=np.float64)
    mean_abs = np.abs(attrs).mean(axis=(0, 1))
    mean_signed = attrs.mean(axis=(0, 1))
    ranking = [
        {
            "sensor": features[j],
            "mean_abs_attribution": float(mean_abs[j]),
            "mean_signed_attribution": float(mean_signed[j]),
        }
        for j in range(len(features))
    ]
    ranking.sort(key=lambda row: row["mean_abs_attribution"], reverse=True)
    per_timestep = np.abs(attrs).mean(axis=(0, 2))
    return {
        "ranking": ranking,
        "per_timestep_mean_abs_attribution": [float(v) for v in per_timestep],
    }


def sample_windows(
    x: np.ndarray, n_windows: int = IG_N_WINDOWS, seed: int = IG_SEED
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic subsample of window indices (sorted, no replacement)."""
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(x), size=min(n_windows, len(x)), replace=False))
    return x[idx], idx
