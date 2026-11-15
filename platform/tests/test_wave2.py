"""Wave-2 tests (Phases 4/6/7 continuation, 2026-11-15).

Covered:
- FD002/FD004 feature derivation keeps the operating settings (the
  six-condition subsets must NOT inherit FD001/FD003's drop list) —
  checked against the saved experiment records, not a re-run.
- The Phase 6 transfer record really is the shipped FD001 model applied
  zero-shot: its feature list must equal ml/artifacts/preprocessing.json
  exactly, and its recorded in-domain references must match the source
  artifacts they cite.
- The Phase 7 SHAP record: ranking is a proper descending ranking over
  exactly the AI4I model features, and its comparison block is present.
- CWRU window-feature extraction on a synthetic series (no download
  needed), and — when the official .mat files are present — the
  adapter's verified recording facts (DE lengths from the CWRU
  catalogue: 97 -> 243,938; 105 -> 121,265; 118 -> 122,571;
  130 -> 121,991 samples).
"""
import json

import numpy as np
import pytest

from conftest import PLATFORM_DIR, RAW, REPO_ROOT

EXP = PLATFORM_DIR / "experiments"


def _record(name):
    path = EXP / name
    if not path.exists():
        pytest.skip(f"experiment not run yet: {name}")
    return json.loads(path.read_text())


@pytest.mark.parametrize("subset", ["fd002", "fd004"])
def test_six_condition_subsets_keep_settings(subset):
    rec = _record(f"cmapss_{subset}__taskB__ridge_flat__s42.json")
    features = rec["preprocessing"]["features"]
    # Six operating conditions -> the settings genuinely vary and the
    # derivation rule keeps them (contrast FD003, where all 3 dropped).
    for setting in ("setting_1", "setting_2", "setting_3"):
        assert setting in features
    dropped = {d["column"] for d in rec["preprocessing"]["dropped_candidates"]}
    assert "setting_1" not in dropped


def test_transfer_record_is_shipped_fd001_model():
    rec = _record("cmapss_fd001_to_fd003__taskB__transfer_lstm__s42.json")
    prep = json.loads(
        (REPO_ROOT / "ml" / "artifacts" / "preprocessing.json").read_text())
    # Zero-shot means: FD001's exact features + FD001's scaler, stated.
    assert rec["preprocessing"]["features"] == prep["features"]
    assert "FD001" in rec["preprocessing"]["scaling"]
    assert rec["timing"]["train_s"] == 0.0  # no training in a transfer run
    # Cited references must match their source artifacts exactly.
    metrics = json.loads(
        (REPO_ROOT / "ml" / "artifacts" / "metrics.json").read_text())
    ref = rec["extra"]["reference_fd001_in_domain"]
    assert ref["rmse"] == metrics["test_last_window_per_engine"]["rmse"]
    fd003 = _record("cmapss_fd003__taskB__lstm__s42.json")
    ref3 = rec["extra"]["reference_fd003_in_domain_lstm"]
    assert ref3["rmse"] == fd003["metrics"]["test_last_window_per_engine"]["rmse"]
    # Transfer must be worse than in-domain FD003 (the reportable result).
    transfer_rmse = rec["metrics"]["test_last_window_per_engine"]["rmse"]
    assert transfer_rmse > ref3["rmse"]


def test_shap_record_ranking():
    rec = _record("ai4i2020__taskA__shap_lightgbm__s42.json")
    ranking = rec["extra"]["shap_mean_abs_ranking"]
    values = [r["mean_abs_shap"] for r in ranking]
    assert values == sorted(values, reverse=True)
    assert {r["feature"] for r in ranking} == set(
        rec["preprocessing"]["features"])
    comp = rec["extra"]["comparison_with_permutation_importance"]
    assert comp["top5_overlap"], "top-5 overlap with permutation must exist"
    assert -1.0 <= comp["spearman_rho_all_features"] <= 1.0
    # The retrained model's metrics must reproduce the wave-1 record.
    lgbm = _record("ai4i2020__taskA__lightgbm__s42.json")
    assert rec["metrics"]["f1"] == pytest.approx(lgbm["metrics"]["f1"],
                                                 abs=1e-6)


def test_cwru_window_features_synthetic():
    from platform_lib.adapters.cwru import FEATURE_COLUMNS, window_features

    rng = np.random.default_rng(42)
    feats = window_features(rng.normal(size=2048))
    assert set(feats) == set(FEATURE_COLUMNS)
    assert all(np.isfinite(v) for v in feats.values())
    assert feats["rms"] > 0 and feats["crest_factor"] >= 1.0
    assert feats["peak_to_peak"] == feats["max"] - feats["min"]


@pytest.mark.skipif(not (RAW / "cwru" / "97.mat").exists(),
                    reason="CWRU official .mat files not downloaded")
def test_cwru_adapter_recordings():
    from platform_lib.adapters.cwru import CWRUAdapter

    df, meta = CWRUAdapter(RAW / "cwru").load()
    assert meta["dataset"] == "cwru"
    assert meta["maintenance_history"] == "absent"
    by_file = {r["file"]: r for r in meta["recordings"]}
    # Verified DE-series lengths (CWRU catalogue + the files themselves).
    assert by_file["97.mat"]["n_samples"] == 243938
    assert by_file["105.mat"]["n_samples"] == 121265
    assert by_file["118.mat"]["n_samples"] == 122571
    assert by_file["130.mat"]["n_samples"] == 121991
    counts = df["label"].value_counts().to_dict()
    assert counts == {"Normal": 119, "InnerRace": 59, "Ball": 59,
                      "OuterRace": 59}
