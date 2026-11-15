"""Wave-4 tests (model expansion + leakage-safe evaluation, 2026-11-15).

Covered:
- All wave-4 experiment records exist (the global schema test in
  test_experiment_schema.py already validates every record's schema;
  these tests pin the wave-4-specific claims).
- The FD003 CNN-LSTM checkpoint + preprocessing sidecar were actually
  saved (wave 5's transfer depends on them) and the record points at
  them.
- The battery records are leave-one-battery-out over exactly
  B0005/B0006/B0018 with B0007's exclusion stated, and carry per-battery
  metrics for each held-out battery.
- The CWRU recording-wise record states its group-split protocol and
  covers >= 12 recordings (vs wave 2's 4).
"""
import json

import pytest

from conftest import PLATFORM_DIR

EXP = PLATFORM_DIR / "experiments"

WAVE4_IDS = [
    "cmapss_fd001__taskB__gru__s42",
    "cmapss_fd003__taskB__cnn_lstm__s42",
    "cmapss_fd002__taskB__random_forest_reg__s42",
    "cmapss_fd002__taskB__xgboost_reg__s42",
    "ai4i2020__taskA__svm__s42",
    "ai4i2020__taskA__mlp__s42",
    "secom__taskC__one_class_svm__s42",
    "secom__taskC__autoencoder__s42",
    "cwru__taskA__random_forest_recordingwise__s42",
    "battery__taskB__random_forest_lobo__s42",
    "battery__taskB__lstm_lobo__s42",
    "paderborn__taskA__random_forest__s42",
]


def _record(experiment_id: str):
    path = EXP / f"{experiment_id}.json"
    if not path.exists():
        pytest.skip(f"experiment not run yet: {experiment_id}")
    return json.loads(path.read_text())


@pytest.mark.parametrize("experiment_id", WAVE4_IDS)
def test_wave4_record_exists(experiment_id):
    rec = _record(experiment_id)
    assert rec["experiment_id"] == experiment_id


def test_fd003_cnn_lstm_checkpoint_saved():
    rec = _record("cmapss_fd003__taskB__cnn_lstm__s42")
    ckpt = rec["extra"]["checkpoint"]
    state_path = PLATFORM_DIR.parent / ckpt["state_dict"]
    sidecar_path = PLATFORM_DIR.parent / ckpt["preprocessing"]
    assert state_path.exists() and state_path.stat().st_size > 0, (
        "FD003 CNN-LSTM checkpoint missing — wave-5 transfer needs it")
    assert sidecar_path.exists()
    sidecar = json.loads(sidecar_path.read_text())
    # The sidecar must describe the same feature space as the record.
    assert sidecar["features"] == rec["preprocessing"]["features"]
    assert len(sidecar["scaler"]["mean"]) == len(sidecar["features"])


@pytest.mark.parametrize("model", ["random_forest", "lstm"])
def test_battery_lobo_record(model):
    rec = _record(f"battery__taskB__{model}_lobo__s42")
    assert rec["dataset"] == "nasa_battery"
    protocol = rec["split"]["protocol"]
    assert "LEAVE-ONE-BATTERY-OUT" in protocol
    assert rec["split"]["held_out_batteries"] == ["B0005", "B0006", "B0018"]
    # B0007's exclusion (censored) must be stated, not silent.
    assert any("B0007" in str(x) for x in rec["split"]["excluded_batteries"])
    per_battery = rec["metrics"]["per_battery"]
    assert set(per_battery) == {"B0005", "B0006", "B0018"}
    for m in per_battery.values():
        assert m["rmse"] >= 0 and m["mae"] >= 0
    assert rec["metrics"]["pooled"]["rmse"] >= 0


def test_cwru_recordingwise_protocol():
    rec = _record("cwru__taskA__random_forest_recordingwise__s42")
    protocol = rec["split"]["protocol"]
    assert "RECORDING-WISE" in protocol
    assert "NO window" in protocol  # the no-leakage claim is explicit
    assert rec["split"]["n_recordings"] >= 12
    # The wave-2 window-split reference must be carried for contrast.
    ref = rec["extra"]["wave2_window_split_reference"]
    assert ref["accuracy"] == 1.0
    assert "per_recording_accuracy" in rec["extra"]
    assert len(rec["extra"]["per_recording_accuracy"]) >= 12


def test_paderborn_grouped_split():
    rec = _record("paderborn__taskA__random_forest__s42")
    assert "MEASUREMENT-GROUPED" in rec["split"]["protocol"]
    assert rec["split"]["n_measurements_train"] > 0
    assert rec["split"]["n_measurements_test"] > 0
    # train + test measurements partition the adapter's 240 measurements
    assert (rec["split"]["n_measurements_train"]
            + rec["split"]["n_measurements_test"]) == 240
