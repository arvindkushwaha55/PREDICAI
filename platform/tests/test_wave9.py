"""Wave-9 tests (frequency-domain bearing features + IMS Tests 2-3,
2026-11-15).

Covered:
- The spectral feature function: exact frequency recovery on a
  synthetic sine, band energies summing to 1, entropy behaviour,
  envelope peak on an amplitude-modulated signal, finiteness, and
  the combined set being the time-domain set plus the spectral set
  (in the documented order).
- The wave-9 records exist and carry their comparison baselines
  (wave-4 for the recording-wise re-run, wave-5 for the transfers) —
  a wave-9 number without its time-domain baseline is not reportable.
- The IMS Test 2/3 records are framed as alarm timing only, with no
  classification-metric keys anywhere (same discipline as wave 5),
  and their snapshot counts match the Readme-documented experiments.
- The IMS adapter loads Tests 2/3 with the 4-channel layout.
"""
import json

import numpy as np
import pytest

from conftest import PLATFORM_DIR, RAW

EXP = PLATFORM_DIR / "experiments"

WAVE9_IDS = [
    "cwru__taskA__random_forest_spectral_recordingwise__s42",
    "cwru_to_paderborn__taskA__transfer_rf_spectral__s42",
    "paderborn_to_cwru__taskA__transfer_rf_spectral__s42",
    "ims_test2__taskC__isolation_forest__s42",
    "ims_test3__taskC__isolation_forest__s42",
]


def _record(experiment_id: str):
    path = EXP / f"{experiment_id}.json"
    if not path.exists():
        pytest.skip(f"experiment not run yet: {experiment_id}")
    return json.loads(path.read_text())


# ------------------------------------------------------- spectral features
def test_spectral_features_on_sine():
    from platform_lib.features.spectral import (SPECTRAL_FEATURE_COLUMNS,
                                                spectral_features)

    n = 2048
    t = np.arange(n)
    # Sine at FFT bin 128 -> exactly 0.125 of Nyquist.
    x = np.sin(2 * np.pi * 128 * t / n)
    f = spectral_features(x)
    assert list(f) == SPECTRAL_FEATURE_COLUMNS
    assert all(np.isfinite(v) for v in f.values())
    assert f["dom_freq_frac"] == pytest.approx(0.125)
    assert f["spectral_centroid_frac"] == pytest.approx(0.125, abs=1e-6)
    assert f["spectral_entropy"] == pytest.approx(0.0, abs=1e-6)
    bands = [f[f"band_{i}"] for i in range(1, 9)]
    assert sum(bands) == pytest.approx(1.0)
    assert max(bands) > 0.99  # one pure tone -> one band holds the energy


def test_spectral_features_entropy_and_envelope():
    from platform_lib.features.spectral import spectral_features

    n = 2048
    t = np.arange(n)
    rng = np.random.default_rng(42)
    noise = spectral_features(rng.normal(size=n))
    assert noise["spectral_entropy"] > 0.9  # white noise ~ flat spectrum
    assert sum(noise[f"band_{i}"] for i in range(1, 9)) == pytest.approx(1.0)

    # Amplitude modulation at bin 16 -> envelope peak at 2*16/2048.
    am = (1 + 0.8 * np.sin(2 * np.pi * 16 * t / n)) \
        * np.sin(2 * np.pi * 256 * t / n)
    f = spectral_features(am)
    assert f["env_peak_freq_frac"] == pytest.approx(2 * 16 / n)
    assert 0.0 <= f["env_peak_amp_norm"] <= 1.0
    assert all(np.isfinite(v) for v in f.values())


def test_combined_features_extend_time_domain():
    from platform_lib.adapters.cwru import FEATURE_COLUMNS
    from platform_lib.features.spectral import (COMBINED_FEATURE_COLUMNS,
                                                combined_window_features)

    assert COMBINED_FEATURE_COLUMNS[:len(FEATURE_COLUMNS)] \
        == FEATURE_COLUMNS
    assert len(COMBINED_FEATURE_COLUMNS) == 24
    rng = np.random.default_rng(42)
    feats = combined_window_features(rng.normal(size=2048))
    assert list(feats) == COMBINED_FEATURE_COLUMNS
    assert all(np.isfinite(v) for v in feats.values())


# ------------------------------------------------------------- records
@pytest.mark.parametrize("experiment_id", WAVE9_IDS)
def test_wave9_record_exists(experiment_id):
    rec = _record(experiment_id)
    assert rec["experiment_id"] == experiment_id


def test_cwru_spectral_record_carries_wave4_comparison():
    rec = _record("cwru__taskA__random_forest_spectral_recordingwise__s42")
    comp = rec["extra"]["comparison_wave4_time_domain"]
    base = comp["wave4_time_domain_only"]
    assert base["accuracy"] == pytest.approx(0.6649, abs=1e-3)
    assert base["f1_macro"] == pytest.approx(0.4214, abs=1e-3)
    assert base["innerrace_correct"] == 0
    assert comp["delta_accuracy"] == pytest.approx(
        rec["metrics"]["accuracy"] - base["accuracy"])
    # Same protocol: 12 recordings, pooled out-of-fold metrics.
    assert rec["split"]["n_recordings"] == 12
    assert rec["split"]["n_windows"] == 1122
    assert "ONLY change is the feature set" in rec["split"]["protocol"]
    assert len(rec["preprocessing"]["features"]) == 24


@pytest.mark.parametrize("experiment_id,src,dst", [
    ("cwru_to_paderborn__taskA__transfer_rf_spectral__s42",
     "cwru", "paderborn"),
    ("paderborn_to_cwru__taskA__transfer_rf_spectral__s42",
     "paderborn", "cwru"),
])
def test_spectral_transfer_carries_wave5_baseline(experiment_id, src, dst):
    rec = _record(experiment_id)
    assert rec["split"]["source_dataset"] == src
    assert rec["split"]["target_dataset"] == dst
    assert "zero-shot" in rec["split"]["protocol"].lower()
    base = rec["extra"]["wave5_time_domain_baseline"]
    assert base["record"] == f"{src}_to_{dst}__taskA__transfer_rf__s42"
    assert base["normal_f1"] == 0.0  # wave-5 finding being re-tested
    assert rec["metrics"]["confusion_labels"] == [
        "Normal", "InnerRace", "OuterRace"]
    assert len(rec["preprocessing"]["features"]) == 24
    assert "fractions of Nyquist" in rec["preprocessing"]["feature_note"]


@pytest.mark.parametrize("experiment_id,n_snapshots,failed", [
    ("ims_test2__taskC__isolation_forest__s42", 984, "bearing_1"),
    ("ims_test3__taskC__isolation_forest__s42", 4448, "bearing_3"),
])
def test_ims_wave9_records_are_alarm_timing(experiment_id, n_snapshots,
                                            failed):
    rec = _record(experiment_id)
    assert "ALARM TIMING" in rec["metrics"]["framing"]
    # No classification metrics may appear anywhere in the record.
    text = json.dumps(rec)
    for forbidden in ('"precision"', '"recall"', '"f1"', '"accuracy"'):
        assert forbidden not in text
    timing = rec["metrics"]["alarm_timing"]
    assert set(timing) == {"bearing_1", "bearing_2", "bearing_3",
                           "bearing_4"}
    assert timing[failed]["documented_defect"] is not None
    others = [b for b in timing if b != failed]
    assert all(timing[b]["documented_defect"] is None for b in others)
    # Snapshot counts are the Readme-documented experiment sizes.
    assert rec["split"]["n_snapshots"] == n_snapshots
    # Identical protocol to wave 5: 30% train window.
    assert rec["split"]["n_train_snapshots_per_bearing"] \
        == int(0.30 * n_snapshots)


def test_ims_adapter_test2_layout(tmp_path):
    from platform_lib.adapters.ims import IMSAdapter

    test_dir = tmp_path / "2nd_test"
    test_dir.mkdir()
    rng = np.random.default_rng(42)
    # Tests 2/3: 4 channels (one accelerometer per bearing).
    np.savetxt(test_dir / "2004.02.12.10.32.39",
               rng.normal(size=(20_480, 4)), delimiter="\t")
    df, meta = IMSAdapter(tmp_path, test="2nd_test").load()
    assert len(df) == 1
    assert meta["channels_per_bearing"] == 1
    assert meta["ground_truth_test2"]["bearing_1"] is not None
    assert meta["ground_truth_test2"]["bearing_3"] is None
    assert "bearing1_ch1_rms" in df.columns
    assert "bearing1_ch2_rms" not in df.columns


def test_ims_adapter_rejects_unknown_test(tmp_path):
    from platform_lib.adapters.ims import IMSAdapter

    with pytest.raises(ValueError):
        IMSAdapter(tmp_path, test="4th_test")
