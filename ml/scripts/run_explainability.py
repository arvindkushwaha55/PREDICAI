"""PREDICAI Part 6 (Phase A5) — Explainability run.

Runs the two primary methods from predicai_ml.explain on the Kaggle-trained
model (ml/artifacts/best_model.pt) against the validation windows:

1. Permutation importance — all 3,272 val windows, seed 123, 1 repeat.
2. Integrated Gradients — 200 val windows (seed-123 subsample), 50 steps,
   zero baseline in normalized space (= train-mean operating point).

Writes ml/artifacts/explainability.json + two bar charts, runs the sanity
checks the report relies on, and prints a summary. Association language
only — see docs/EXPLAINABILITY.md.

Run:  python ml/scripts/run_explainability.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from predicai_ml.data import load_fd001  # noqa: E402
from predicai_ml.explain import (  # noqa: E402
    IG_N_WINDOWS, IG_SEED, IG_STEPS, integrated_gradients,
    permutation_importance, sample_windows, summarize_ig,
)
from predicai_ml.model import LSTMRULRegressor  # noqa: E402
from predicai_ml.preprocessing import FEATURES, run_pipeline  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "ml" / "data" / "raw"
ARTIFACTS = ROOT / "ml" / "artifacts"
WEIGHTS = ARTIFACTS / "best_model.pt"
METRICS = ARTIFACTS / "metrics.json"
OUT_JSON = ARTIFACTS / "explainability.json"
OUT_PERM_PNG = ARTIFACTS / "permutation_importance.png"
OUT_IG_PNG = ARTIFACTS / "integrated_gradients.png"

HONESTY_NOTE = (
    "Attributions describe ASSOCIATION: which sensors the trained model "
    "relies on when it predicts RUL. They do not show that a sensor causes "
    "degradation or failure. Permutation importance measures reliance for "
    "accuracy on the validation split; Integrated Gradients measures how "
    "each sensor's deviation from the train-mean operating point moves an "
    "individual prediction. Sensors are correlated in C-MAPSS, so credit "
    "can be shared or shifted between correlated sensors."
)


def bar_chart(rows: list[dict], value_key: str, xlabel: str, title: str,
              out: Path) -> None:
    """Horizontal bar chart, most important sensor on top."""
    labels = [r["sensor"] for r in rows][::-1]
    values = [r[value_key] for r in rows][::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(labels, values, color="#2b6cb0")
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    metrics = json.loads(METRICS.read_text())
    cfg = metrics["config"]

    train, test, rul = load_fd001(RAW)
    res = run_pipeline(train, test, rul)

    model = LSTMRULRegressor(
        len(FEATURES), cfg["hidden_1"], cfg["hidden_2"],
        cfg["head_dim"], cfg["dropout"],
    )
    model.load_state_dict(
        torch.load(WEIGHTS, map_location="cpu", weights_only=True))
    model.eval()

    # ------------------------------------------------ 1. permutation
    perm = permutation_importance(model, res.X_val, res.y_val, FEATURES)
    perm_rows = [
        {"sensor": r["sensor"], "rmse_baseline": perm["rmse_baseline"],
         "rmse_permuted": r["rmse_permuted"],
         "rmse_increase": r["rmse_increase"]}
        for r in perm["ranking"]
    ]

    # ------------------------------------------------ 2. integrated gradients
    x_ig, idx_ig = sample_windows(res.X_val, IG_N_WINDOWS, IG_SEED)
    ig = integrated_gradients(model, x_ig, steps=IG_STEPS)
    ig_summary = summarize_ig(ig.attributions, FEATURES)

    # ------------------------------------------------ sanity checks
    expected_val_rmse = metrics["val_all_windows"]["rmse"]
    baseline_diff = abs(perm["rmse_baseline"] - expected_val_rmse)
    perm_sensors = {r["sensor"] for r in perm_rows}
    ig_sensors = {r["sensor"] for r in ig_summary["ranking"]}
    nan_free = bool(
        np.isfinite([r["rmse_increase"] for r in perm_rows]).all()
        and np.isfinite(
            [r["mean_abs_attribution"] for r in ig_summary["ranking"]]).all()
        and np.isfinite(ig.attributions).all()
    )
    top5_perm = {r["sensor"] for r in perm_rows[:5]}
    top5_ig = {r["sensor"] for r in ig_summary["ranking"][:5]}
    sanity = {
        "permutation_baseline_rmse": perm["rmse_baseline"],
        "metrics_json_val_rmse": expected_val_rmse,
        "baseline_abs_diff": baseline_diff,
        "baseline_matches_metrics_json": bool(baseline_diff < 0.1),
        "ig_completeness_mean_abs_error_cycles":
            ig.completeness_mean_abs_error,
        "ig_completeness_mean_rel_error": ig.completeness_mean_rel_error,
        "sensors_in_permutation_ranking": len(perm_sensors),
        "sensors_in_ig_ranking": len(ig_sensors),
        "all_14_sensors_in_both_rankings":
            perm_sensors == ig_sensors == set(FEATURES),
        "no_nans": nan_free,
        "top5_overlap_between_methods": sorted(top5_perm & top5_ig),
    }
    assert sanity["baseline_matches_metrics_json"], "baseline RMSE drifted"
    assert sanity["all_14_sensors_in_both_rankings"], "sensor coverage broken"
    assert sanity["no_nans"], "NaN/Inf in explainability outputs"

    # ------------------------------------------------ artifact
    payload = {
        "part": "PREDICAI Part 6 — Explainability (Phase A5)",
        "model": {
            "weights": "ml/artifacts/best_model.pt",
            "architecture": "LSTMRULRegressor(14 -> 128 -> 64 -> head 32)",
            "trained_on": "Kaggle (owner run); see metrics.json",
            "test_rmse_cycles": metrics["test_last_window_per_engine"]["rmse"],
        },
        "data": {
            "split": "validation — engine-wise holdout, all windows",
            "n_val_windows": int(len(res.X_val)),
            "features": FEATURES,
        },
        "honesty_note": HONESTY_NOTE,
        "methods": {
            "permutation_importance": perm["params"],
            "integrated_gradients": {
                "seed": IG_SEED,
                "n_windows": int(len(x_ig)),
                "ig_steps": IG_STEPS,
                "baseline": (
                    "zero vector in normalized (z-score) space = the "
                    "train-split mean of every sensor, i.e. the average "
                    "engine's operating point; the model's raw prediction "
                    "at that baseline is reported below"
                ),
                "baseline_prediction_cycles": ig.baseline_prediction,
                "attribution_units": (
                    "cycles of predicted RUL (attributions sum to "
                    "prediction minus baseline prediction)"
                ),
            },
        },
        "permutation_importance": {
            "rmse_baseline": perm["rmse_baseline"],
            "ranking": perm_rows,
        },
        "integrated_gradients": {
            "ranking": ig_summary["ranking"],
            "per_timestep_mean_abs_attribution":
                ig_summary["per_timestep_mean_abs_attribution"],
            "mean_prediction_on_sampled_windows": float(
                np.mean(ig.predictions)),
        },
        "sanity_checks": sanity,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2))

    bar_chart(
        perm_rows, "rmse_increase",
        "RMSE increase when the sensor is shuffled (cycles)",
        "Permutation importance — FD001 validation windows",
        OUT_PERM_PNG,
    )
    bar_chart(
        ig_summary["ranking"], "mean_abs_attribution",
        "Mean |attribution| (cycles of predicted RUL)",
        "Integrated Gradients — FD001 validation windows (n=200)",
        OUT_IG_PNG,
    )

    print(f"Permutation baseline RMSE: {perm['rmse_baseline']:.4f} "
          f"(metrics.json val RMSE: {expected_val_rmse:.4f}, "
          f"diff {baseline_diff:.6f})")
    print("\nPermutation ranking (RMSE increase, cycles):")
    for r in perm_rows:
        print(f"  {r['sensor']:>10}  +{r['rmse_increase']:6.3f}  "
              f"(permuted RMSE {r['rmse_permuted']:.3f})")
    print(f"\nIG baseline prediction (zero window): "
          f"{ig.baseline_prediction:.3f} cycles")
    print(f"IG completeness: mean abs error "
          f"{ig.completeness_mean_abs_error:.4f} cycles, "
          f"mean rel error {ig.completeness_mean_rel_error:.4%}")
    print("\nIG ranking (mean |attribution|, cycles):")
    for r in ig_summary["ranking"]:
        print(f"  {r['sensor']:>10}  {r['mean_abs_attribution']:7.4f}  "
              f"(signed {r['mean_signed_attribution']:+.4f})")
    print(f"\nTop-5 overlap between methods: "
          f"{sanity['top5_overlap_between_methods']}")
    print(f"\nWrote {OUT_JSON}\nWrote {OUT_PERM_PNG}\nWrote {OUT_IG_PNG}")


if __name__ == "__main__":
    main()
