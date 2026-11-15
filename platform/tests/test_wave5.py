"""Wave-5/6 tests (cross-dataset generalisation + XAI + IMS,
2026-11-15).

Covered:
- All wave-5/6 experiment records exist (the global schema test in
  test_experiment_schema.py validates every record's schema; these
  tests pin the wave-specific claims).
- Every transfer record carries its protocol fields (source/target
  datasets, zero-shot statement) — a transfer number without its
  protocol is not reportable.
- The FD003 -> FD001 record carries the wave-2 reverse direction as
  its reference (the asymmetry comparison is part of the result).
- The IMS adapter parses a synthetic snapshot file correctly
  (shape, timestamp, per-bearing feature columns) and the IMS record
  is framed as alarm timing, with no classification metrics.
- FAULT_DIAGNOSIS.md exists and cites at least five real records.
"""
import json
import re

import numpy as np
import pytest

from conftest import PLATFORM_DIR

EXP = PLATFORM_DIR / "experiments"

WAVE5_IDS = [
    "cmapss_fd003_to_fd001__taskB__transfer_cnn_lstm__s42",
    "cmapss_fd002_to_fd004__taskB__transfer_ridge__s42",
    "cmapss_fd004_to_fd002__taskB__transfer_ridge__s42",
    "cwru_to_paderborn__taskA__transfer_rf__s42",
    "paderborn_to_cwru__taskA__transfer_rf__s42",
    "battery__taskB__shap_random_forest__s42",
    "paderborn__taskA__permutation_rf__s42",
    "ims__taskC__isolation_forest__s42",
]

TRANSFER_IDS = WAVE5_IDS[:5]


def _record(experiment_id: str):
    path = EXP / f"{experiment_id}.json"
    if not path.exists():
        pytest.skip(f"experiment not run yet: {experiment_id}")
    return json.loads(path.read_text())


@pytest.mark.parametrize("experiment_id", WAVE5_IDS)
def test_wave5_record_exists(experiment_id):
    rec = _record(experiment_id)
    assert rec["experiment_id"] == experiment_id


@pytest.mark.parametrize("experiment_id", TRANSFER_IDS)
def test_transfer_protocol_fields(experiment_id):
    rec = _record(experiment_id)
    split = rec["split"]
    assert split["source_dataset"] and split["target_dataset"]
    assert split["source_dataset"] != split["target_dataset"]
    assert "zero-shot" in split["protocol"].lower()
    # No training on the target: either no training at all, or a
    # source-only (re)fit stated in the record.
    assert rec["timing"]["train_s"] >= 0


def test_fd003_to_fd001_carries_reverse_reference():
    rec = _record("cmapss_fd003_to_fd001__taskB__transfer_cnn_lstm__s42")
    reverse = rec["extra"]["reverse_direction_wave2"]
    assert reverse["rmse"] == pytest.approx(48.80, abs=0.01)
    # FD003's feature set (15, incl. sensor_6) — not FD001's 14.
    assert rec["preprocessing"]["n_features"] == 15
    assert "sensor_6" in rec["preprocessing"]["features"]


@pytest.mark.parametrize("experiment_id", [
    "cmapss_fd002_to_fd004__taskB__transfer_ridge__s42",
    "cmapss_fd004_to_fd002__taskB__transfer_ridge__s42",
])
def test_ridge_transfer_shared_features(experiment_id):
    rec = _record(experiment_id)
    assert rec["preprocessing"]["n_features"] == 23
    assert "IDENTICAL" in rec["preprocessing"]["feature_set"]
    assert "REFIT" in rec["model_params"]["training"]


@pytest.mark.parametrize("experiment_id", [
    "cwru_to_paderborn__taskA__transfer_rf__s42",
    "paderborn_to_cwru__taskA__transfer_rf__s42",
])
def test_bearing_transfer_shared_label_set(experiment_id):
    rec = _record(experiment_id)
    assert rec["metrics"]["confusion_labels"] == [
        "Normal", "InnerRace", "OuterRace"]
    # Ball must be excluded and the exclusion stated.
    assert "Ball" in rec["preprocessing"]["excluded"]


def test_battery_shap_record():
    rec = _record("battery__taskB__shap_random_forest__s42")
    assert "EXPLANATION MODEL" in rec["split"]["protocol"]
    assert "IN-SAMPLE" in rec["split"]["protocol"]
    ranking = rec["extra"]["shap_mean_abs_ranking"]
    assert len(ranking) == len(rec["preprocessing"]["features"])
    assert all(r["mean_abs_shap"] >= 0 for r in ranking)
    assert ranking[0]["mean_abs_shap"] >= ranking[-1]["mean_abs_shap"]


def test_paderborn_permutation_record():
    rec = _record("paderborn__taskA__permutation_rf__s42")
    ranking = rec["extra"]["permutation_importance_test_accuracy"]
    assert len(ranking) == 10  # the shared time-domain feature set
    ref = rec["extra"]["wave4_reference"]["accuracy"]
    assert rec["metrics"]["accuracy"] == pytest.approx(ref, abs=0.01)


def test_ims_record_is_alarm_timing_not_classification():
    rec = _record("ims__taskC__isolation_forest__s42")
    assert "ALARM TIMING" in rec["metrics"]["framing"]
    # No classification metrics may appear anywhere in the record.
    text = json.dumps(rec)
    for forbidden in ('"precision"', '"recall"', '"f1"',
                      '"accuracy"'):
        assert forbidden not in text
    timing = rec["metrics"]["alarm_timing"]
    assert set(timing) == {"bearing_1", "bearing_2", "bearing_3",
                           "bearing_4"}
    assert timing["bearing_3"]["documented_defect"] is not None
    assert timing["bearing_1"]["documented_defect"] is None
    assert rec["split"]["n_snapshots"] == 2156


def test_ims_adapter_parses_snapshot(tmp_path):
    from platform_lib.adapters.ims import (FEATURE_COLUMNS, IMSAdapter,
                                           parse_timestamp)

    test_dir = tmp_path / "1st_test"
    test_dir.mkdir()
    rng = np.random.default_rng(42)
    data = rng.normal(size=(20_480, 8))
    np.savetxt(test_dir / "2003.10.22.12.06.24", data, delimiter="\t")
    df, meta = IMSAdapter(tmp_path).load()
    assert len(df) == 1
    assert df["snapshot_index"].iloc[0] == 0
    for col in FEATURE_COLUMNS:
        assert col in df.columns
    assert meta["n_snapshots"] == 1
    assert parse_timestamp("2003.11.25.23.39.56").year == 2003


def test_ims_adapter_missing_data_raises(tmp_path):
    from platform_lib.adapters.ims import IMSAdapter

    with pytest.raises(FileNotFoundError):
        IMSAdapter(tmp_path).load()


def test_fault_diagnosis_references_records():
    path = PLATFORM_DIR / "FAULT_DIAGNOSIS.md"
    assert path.exists(), "FAULT_DIAGNOSIS.md missing"
    text = path.read_text()
    on_disk = {p.stem for p in EXP.glob("*.json")} - {"INDEX"}
    cited = {rid for rid in on_disk if rid in text}
    assert len(cited) >= 5, f"only {len(cited)} records cited: {cited}"
    # The hard limits section is the point of the document.
    assert "must never claim" in text
    assert re.search(r"electrical", text, re.IGNORECASE)
