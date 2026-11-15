"""Wave-12 tests (FD004 per-regime LSTM + order-domain bearing
features, 2026-11-15).

Covered:
- The characteristic orders computed from bearing geometry match
  the published multipliers (CWRU Data Center's own table for the
  6205; the nominal-geometry values cited for the Paderborn 6203),
  and the `bsf` convention is 2*ball-spin (ball-defect rate).
- The order features respond to physics, not noise: a synthetic
  amplitude-modulated signal modulated AT the BPFO order puts its
  envelope energy in the BPFO band (and not in the BPFI band),
  symmetrically for BPFI, and an unmodulated carrier puts ~none in
  either.
- The FD004 per-regime LSTM record carries BOTH comparison blocks
  (wave-2 global-normalisation LSTM; wave-10 per-regime Ridge)
  with the baselines as recorded in those records and deltas that
  match its own metrics.
- The three bearing records carry their wave-5 (time-domain) and
  wave-9 (spectral) baselines with consistent deltas, the 19-feature
  combined set, and the order-physics documentation block.
"""
import json

import numpy as np
import pytest

from conftest import PLATFORM_DIR

EXP = PLATFORM_DIR / "experiments"

FD004_LSTM_ID = "cmapss_fd004__taskB__lstm_perregime__s42"
CWRU_ORDER_ID = "cwru__taskA__random_forest_order_recordingwise__s42"
TRANSFER_IDS = [
    "cwru_to_paderborn__taskA__transfer_rf_order__s42",
    "paderborn_to_cwru__taskA__transfer_rf_order__s42",
]

# Baselines as recorded in the earlier waves' own records.
WAVE2_FD004_LSTM_RMSE = 18.8094
WAVE10_FD004_RIDGE_PERREGIME_RMSE = 19.1146
WAVE4_CWRU_ACC = 0.6649
WAVE9_CWRU_ACC = 0.6791
WAVE5_TRANSFER_ACC = {
    "cwru_to_paderborn__taskA__transfer_rf_order__s42": 0.2531,
    "paderborn_to_cwru__taskA__transfer_rf_order__s42": 0.1249,
}
WAVE9_TRANSFER_ACC = {
    "cwru_to_paderborn__taskA__transfer_rf_order__s42": 0.2102,
    "paderborn_to_cwru__taskA__transfer_rf_order__s42": 0.1873,
}


def _record(experiment_id: str):
    path = EXP / f"{experiment_id}.json"
    if not path.exists():
        pytest.skip(f"experiment not run yet: {experiment_id}")
    return json.loads(path.read_text())


# ------------------------------------------------------------- physics
def test_cwru_orders_match_published_multipliers():
    from platform_lib.features.order import CWRU_6205

    o = CWRU_6205.fault_orders()
    pub = CWRU_6205.published_multipliers
    assert o["bpfo"] == pytest.approx(pub["bpfo"], abs=1e-3)
    assert o["bpfi"] == pytest.approx(pub["bpfi"], abs=1e-3)
    assert o["ftf"] == pytest.approx(pub["ftf"], abs=1e-3)
    # The published rolling-element multiplier is the ball-DEFECT
    # rate 2*BSF, and the module's `bsf` follows that convention.
    assert o["bsf"] == pytest.approx(pub["bsf_defect_2xbsf"], abs=1e-3)
    assert o["bsf"] == pytest.approx(2.0 * o["bsf_spin"])


def test_paderborn_orders_match_nominal_geometry_values():
    from platform_lib.features.order import PADERBORN_6203

    o = PADERBORN_6203.fault_orders()
    pub = PADERBORN_6203.published_multipliers
    assert o["bpfo"] == pytest.approx(pub["bpfo"], abs=1e-3)
    assert o["bpfi"] == pytest.approx(pub["bpfi"], abs=1e-3)
    assert o["bsf_spin"] == pytest.approx(pub["bsf_spin"], abs=1e-3)
    assert o["bsf"] == pytest.approx(2.0 * o["bsf_spin"])


# ------------------------------------------------- synthetic AM signal
def _am_signal(mod_order: float | None, seed: int = 1):
    """3 kHz carrier amplitude-modulated at `mod_order` x shaft
    (30 Hz at 1800 rpm), 2048 samples at 12 kHz — the CWRU window."""
    from platform_lib.features.order import CWRU_6205

    fs, rpm, n = 12_000.0, 1800.0, 2048
    t = np.arange(n) / fs
    rng = np.random.default_rng(seed)
    carrier = np.sin(2 * np.pi * 3000 * t)
    if mod_order is None:
        sig = carrier
    else:
        sig = (1 + 0.9 * np.cos(2 * np.pi * mod_order * 30.0 * t)) \
            * carrier
    return sig + 0.05 * rng.standard_normal(n), fs, rpm, CWRU_6205


def test_order_feature_peaks_at_modulated_order():
    from platform_lib.features.order import order_features

    orders = _am_signal(None)[3].fault_orders()
    sig_bpfo, fs, rpm, spec = _am_signal(orders["bpfo"])
    sig_bpfi, _, _, _ = _am_signal(orders["bpfi"])
    sig_none, _, _, _ = _am_signal(None)

    f_bpfo = order_features(sig_bpfo, fs, rpm, spec)
    f_bpfi = order_features(sig_bpfi, fs, rpm, spec)
    f_none = order_features(sig_none, fs, rpm, spec)

    # Modulated at BPFO: the BPFO band carries the envelope energy,
    # the BPFI band ~none — and vice versa for the BPFI control.
    assert f_bpfo["ord_bpfo"] > 0.1
    assert f_bpfo["ord_bpfo"] > 10 * f_bpfo["ord_bpfi"]
    assert f_bpfi["ord_bpfi"] > 0.1
    assert f_bpfi["ord_bpfi"] > 10 * f_bpfi["ord_bpfo"]
    # Unmodulated control: neither band carries anything.
    assert f_none["ord_bpfo"] < 0.01
    assert f_none["ord_bpfi"] < 0.01
    # All features finite and in range on every signal.
    for feats in (f_bpfo, f_bpfi, f_none):
        assert all(np.isfinite(v) for v in feats.values())
        assert all(0.0 <= v <= 1.0 for v in feats.values())


def test_order_features_silent_window_is_zero():
    from platform_lib.features.order import CWRU_6205, order_features

    feats = order_features(np.ones(2048), 12_000.0, 1800.0, CWRU_6205)
    assert all(v == 0.0 for v in feats.values())


# ---------------------------------------------------- FD004 LSTM record
def test_fd004_perregime_lstm_record_comparisons():
    rec = _record(FD004_LSTM_ID)
    new = rec["metrics"]["test_last_window_per_engine"]

    comp = rec["extra"]["comparison_wave2_global_normalisation"]
    assert comp["wave2_global"]["rmse"] == pytest.approx(
        WAVE2_FD004_LSTM_RMSE, abs=1e-3)
    assert comp["wave12_perregime"]["rmse"] == pytest.approx(new["rmse"])
    assert comp["delta_rmse"] == pytest.approx(
        new["rmse"] - comp["wave2_global"]["rmse"])

    comp_r = rec["extra"]["comparison_wave10_perregime_ridge"]
    assert comp_r["wave10_ridge_perregime"]["rmse"] == pytest.approx(
        WAVE10_FD004_RIDGE_PERREGIME_RMSE, abs=1e-3)
    assert comp_r["wave12_lstm_perregime"]["rmse"] == pytest.approx(
        new["rmse"])
    assert comp_r["delta_rmse"] == pytest.approx(
        new["rmse"] - comp_r["wave10_ridge_perregime"]["rmse"])

    # Same wave-10 per-regime protocol: 6 regimes, train-only fit.
    regimes = rec["preprocessing"]["regimes"]
    assert regimes["k"] == 6
    assert len(regimes["train_sizes_rows"]) == 6
    assert "ONLY change vs wave 2" in rec["split"]["protocol"]
    assert "PER-REGIME" in rec["preprocessing"]["scaling"]


# ------------------------------------------------------- bearing records
def test_cwru_order_record_carries_both_baselines():
    rec = _record(CWRU_ORDER_ID)
    assert len(rec["preprocessing"]["features"]) == 19
    comp = rec["extra"]["comparison_previous_feature_sets"]
    assert comp["wave4_time_domain_only"]["accuracy"] == pytest.approx(
        WAVE4_CWRU_ACC, abs=1e-3)
    assert comp["wave9_combined_spectral"]["accuracy"] == pytest.approx(
        WAVE9_CWRU_ACC, abs=1e-3)
    acc = rec["metrics"]["accuracy"]
    assert comp["delta_accuracy_vs_wave4"] == pytest.approx(
        acc - comp["wave4_time_domain_only"]["accuracy"])
    assert comp["delta_accuracy_vs_wave9"] == pytest.approx(
        acc - comp["wave9_combined_spectral"]["accuracy"])
    phys = rec["extra"]["order_physics_cwru"]
    assert phys["computed_orders"]["bpfo"] == pytest.approx(
        3.5848, abs=1e-3)
    # Per-recording speeds are documented, with their source.
    recs = rec["preprocessing"]["recordings"]
    assert len(recs) == 12
    assert all("rpm_source" in r for r in recs)


@pytest.mark.parametrize("experiment_id", TRANSFER_IDS)
def test_transfer_order_record_carries_baselines(experiment_id):
    rec = _record(experiment_id)
    assert len(rec["preprocessing"]["features"]) == 19
    acc = rec["metrics"]["accuracy"]

    w5 = rec["extra"]["wave5_time_domain_baseline"]
    assert w5["accuracy"] == pytest.approx(
        WAVE5_TRANSFER_ACC[experiment_id], abs=1e-3)
    assert rec["extra"]["delta_vs_wave5_time_domain"][
        "accuracy_delta"] == pytest.approx(acc - w5["accuracy"])

    w9 = rec["extra"]["wave9_spectral_baseline"]
    assert w9["accuracy"] == pytest.approx(
        WAVE9_TRANSFER_ACC[experiment_id], abs=1e-3)
    assert rec["extra"]["delta_vs_wave9_spectral"][
        "accuracy_delta"] == pytest.approx(acc - w9["accuracy"])

    phys = rec["extra"]["order_physics"]
    assert phys["cwru"]["computed_orders"]["bpfi"] == pytest.approx(
        5.4152, abs=1e-3)
    assert phys["paderborn"]["computed_orders"]["bpfi"] == \
        pytest.approx(4.9457, abs=1e-3)
    assert "2*BSF" in phys["cwru"]["bsf_convention"] or \
        "2*bsf" in phys["cwru"]["bsf_convention"].lower()
