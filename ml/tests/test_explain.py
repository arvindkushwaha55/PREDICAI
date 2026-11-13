"""PREDICAI — explainability tests (Part 6). Tiny model, synthetic windows.

Checks the invariants the Part 6 report relies on:
- permutation ranking covers every feature, sorted, and is deterministic
  for a fixed seed;
- IG attributions have the input's shape and satisfy completeness
  (sum of attributions ~= f(x) - f(0)) on a smooth small model;
- the IG summary covers every feature.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from predicai_ml.explain import (  # noqa: E402
    integrated_gradients, permutation_importance, summarize_ig,
)
from predicai_ml.model import LSTMRULRegressor  # noqa: E402

FEATURES = ["sensor_a", "sensor_b", "sensor_c"]


def tiny_setup(n: int = 24, seq: int = 6, seed: int = 0):
    torch.manual_seed(seed)
    model = LSTMRULRegressor(len(FEATURES), hidden_1=16, hidden_2=8,
                             head_dim=8, dropout=0.2)
    model.eval()
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, seq, len(FEATURES))).astype(np.float32)
    y = (100.0 - 5.0 * x[:, -1, 0]).astype(np.float32)
    return model, x, y


def test_permutation_importance_shape_and_determinism():
    model, x, y = tiny_setup()
    out1 = permutation_importance(model, x, y, FEATURES, seed=123)
    out2 = permutation_importance(model, x, y, FEATURES, seed=123)
    assert out1["rmse_baseline"] > 0
    assert [r["sensor"] for r in out1["ranking"]] == \
        [r["sensor"] for r in out2["ranking"]]
    assert {r["sensor"] for r in out1["ranking"]} == set(FEATURES)
    increases = [r["rmse_increase"] for r in out1["ranking"]]
    assert increases == sorted(increases, reverse=True)


def test_integrated_gradients_completeness():
    model, x, _ = tiny_setup()
    result = integrated_gradients(model, x[:8], steps=50)
    assert result.attributions.shape == x[:8].shape
    assert np.isfinite(result.attributions).all()
    # Completeness on a smooth model with 50 midpoint steps: well under 5%.
    assert result.completeness_mean_rel_error < 0.05


def test_summarize_ig_covers_all_features():
    model, x, _ = tiny_setup()
    result = integrated_gradients(model, x[:8], steps=10)
    summary = summarize_ig(result.attributions, FEATURES)
    assert {r["sensor"] for r in summary["ranking"]} == set(FEATURES)
    assert len(summary["per_timestep_mean_abs_attribution"]) == x.shape[1]
