"""Wave-10 tests (per-regime normalisation + Paderborn real damage,
2026-11-15).

Covered:
- Regime derivation is TRAIN-ONLY: the RegimeModel's sizes/centroids
  come exclusively from the frame passed to derive_regimes, and
  assignment of new rows uses only the stored train-fitted centroids
  (nearest-centroid in standardised setting space), never the new
  rows' own statistics.
- The per-regime pipeline on FD002 reproduces the wave-2 feature set
  and split sizes exactly (only the scaling may differ).
- The three per-regime records carry the wave-2 global-normalisation
  comparison block with the wave-2 numbers as recorded in wave 2's
  own records, and deltas that match the records' own metrics.
- The Paderborn real-damage record documents each bearing's damage
  origin/method from the official damage-profile PDFs (KA04/KI04:
  fatigue pitting, lifetime test), including KI04's secondary damage.
"""
import json

import numpy as np
import pandas as pd
import pytest

from conftest import PLATFORM_DIR, RAW

EXP = PLATFORM_DIR / "experiments"

PERREGIME_IDS = [
    "cmapss_fd002__taskB__ridge_perregime__s42",
    "cmapss_fd004__taskB__ridge_perregime__s42",
    "cmapss_fd002__taskB__lstm_perregime__s42",
]

# Wave-2 global-normalisation test RMSEs, as recorded in wave 2's own
# records (the comparison baseline every wave-10 record must carry).
WAVE2_RMSE = {
    "cmapss_fd002__taskB__ridge_perregime__s42": 16.5817,
    "cmapss_fd004__taskB__ridge_perregime__s42": 19.4250,
    "cmapss_fd002__taskB__lstm_perregime__s42": 16.2363,
}


def _record(experiment_id: str):
    path = EXP / f"{experiment_id}.json"
    if not path.exists():
        pytest.skip(f"experiment not run yet: {experiment_id}")
    return json.loads(path.read_text())


# ------------------------------------------------------ regime fitting
def _synthetic_settings(centres, n_per=60, jitter=0.01, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for c in centres:
        rows.append(np.asarray(c) + rng.normal(0, jitter, size=(n_per, 3)))
    S = np.vstack(rows)
    return pd.DataFrame(S, columns=["setting_1", "setting_2", "setting_3"])


def test_regime_derivation_uses_train_frame_only():
    from platform_lib.regime_pipeline import derive_regimes

    centres = [(0.0, 0.0, 100.0), (40.0, 0.8, 60.0)]
    train = _synthetic_settings(centres, n_per=60)
    model, labels = derive_regimes(train, k=2, seed=42)

    # Sizes account for exactly the train rows — nothing else was fit.
    assert sum(model.train_sizes) == len(train)
    assert sorted(model.train_sizes) == [60, 60]
    # Centroids recover the two planted centres (train statistics only).
    cents = sorted(map(tuple, np.round(model.centroids_raw, 1)))
    assert cents == [(0.0, 0.0, 100.0), (40.0, 0.8, 60.0)]
    # Re-assigning the train rows reproduces the fit's own partition:
    # each assigned group is centred on its stored centroid.
    assigned = model.assign(train)
    for r in range(2):
        grp = train.to_numpy()[assigned == r]
        assert len(grp) == model.train_sizes[r]
        assert np.allclose(grp.mean(axis=0), model.centroids_raw[r],
                           atol=0.05)


def test_regime_assignment_uses_stored_centroids_not_new_data():
    from platform_lib.regime_pipeline import derive_regimes

    centres = [(0.0, 0.0, 100.0), (40.0, 0.8, 60.0)]
    train = _synthetic_settings(centres, n_per=60)
    model, _ = derive_regimes(train, k=2, seed=42)

    # A probe frame sitting exactly on centre 0 must be assigned to
    # centre 0's regime even though the probe frame's OWN mean is
    # nowhere near the train distribution — assignment is a pure
    # function of the stored train-fitted parameters.
    probe = pd.DataFrame([centres[0]] * 5,
                         columns=["setting_1", "setting_2", "setting_3"])
    expected = int(np.argmin(
        ((model.centroids_raw - np.asarray(centres[0])) ** 2).sum(axis=1)))
    assert set(model.assign(probe)) == {expected}
    probe2 = pd.DataFrame([centres[1]] * 5,
                          columns=["setting_1", "setting_2", "setting_3"])
    assert set(model.assign(probe2)) == {1 - expected}


def test_perregime_pipeline_matches_wave2_features_and_split():
    from platform_lib.adapters.cmapss import CMapssAdapter
    from platform_lib.regime_pipeline import build_rul_data_perregime

    adapter = CMapssAdapter(RAW / "cmapss" / "extracted", subset="FD002")
    train, _ = adapter.load()
    test, rul = adapter.load_test()
    data = build_rul_data_perregime(train, test, rul, cap=125, seq_len=30,
                                    seed=42)
    wave2 = _record("cmapss_fd002__taskB__ridge_flat__s42")
    # Same feature set, same split sizes as wave 2 — only scaling moved.
    assert data.features == wave2["preprocessing"]["features"]
    assert len(data.features) == 23
    assert len(data.train_engines) == wave2["split"]["n_train_engines"]
    assert len(data.val_engines) == wave2["split"]["n_val_engines"]
    assert len(data.X_train) == wave2["split"]["n_train_windows"]
    assert data.X_train.shape[1:] == (30, 23)
    assert sum(data.regime_info["train_sizes"]) == 43083  # FD002 train rows
    assert len(data.regime_info["train_sizes"]) == 6
    cents = sorted(tuple(round(v, 2) for v in c.values())
                   for c in data.regime_info["centroids_raw_settings"])
    assert cents == [(0.0, 0.0, 100.0), (10.0, 0.25, 100.0),
                     (20.0, 0.7, 100.0), (25.0, 0.62, 60.0),
                     (35.0, 0.84, 100.0), (42.0, 0.84, 100.0)]


# ------------------------------------------------------------- records
@pytest.mark.parametrize("experiment_id", PERREGIME_IDS)
def test_perregime_record_exists(experiment_id):
    rec = _record(experiment_id)
    assert rec["experiment_id"] == experiment_id


@pytest.mark.parametrize("experiment_id", PERREGIME_IDS)
def test_perregime_record_carries_wave2_comparison(experiment_id):
    rec = _record(experiment_id)
    comp = rec["extra"]["comparison_wave2_global_normalisation"]
    base = comp["wave2_global"]
    assert base["rmse"] == pytest.approx(WAVE2_RMSE[experiment_id],
                                         abs=1e-3)
    new = rec["metrics"]["test_last_window_per_engine"]
    assert comp["wave10_perregime"]["rmse"] == pytest.approx(new["rmse"])
    assert comp["delta_rmse"] == pytest.approx(new["rmse"] - base["rmse"])
    assert comp["delta_nasa"] == pytest.approx(
        new["nasa_score"] - base["nasa_score"])
    # The regimes block: 6 regimes, train rows accounted for, and the
    # protocol states the ONLY change vs wave 2 is the normalisation.
    regimes = rec["preprocessing"]["regimes"]
    assert regimes["k"] == 6
    assert len(regimes["train_sizes_rows"]) == 6
    assert all(s > 0 for s in regimes["train_sizes_rows"])
    assert len(regimes["centroids_raw_settings"]) == 6
    assert "ONLY change vs wave 2" in rec["split"]["protocol"]
    assert "PER-REGIME" in rec["preprocessing"]["scaling"]


def test_paderborn_real_record_documents_damage_origin():
    rec = _record("paderborn_real__taskA__random_forest__s42")
    docs = {d["bearing"]: d
            for d in rec["extra"]["bearing_documentation"]}
    assert set(docs) == {"K001", "KA04", "KI04"}
    assert docs["K001"]["documented_damage"] is None

    ka04 = docs["KA04"]["documented_damage"]
    assert ka04["mode"] == "fatigue"
    assert ka04["symptom"] == "Pitting"
    assert ka04["component"].startswith("OR")
    assert ka04["damage_method"] == "lifetime test"
    assert "real" in docs["KA04"]["damage_origin"]

    ki04 = docs["KI04"]["documented_damage"]
    primary = ki04["damage_1_primary"]
    assert primary["mode"].startswith("fatigue")
    assert primary["symptom"] == "Pitting"
    assert primary["component"].startswith("IR")
    assert primary["damage_method"] == "lifetime test"
    # The secondary outer-race damage is documented, not hidden.
    assert ki04["damage_2_secondary"]["component"].startswith("OR")
    assert "real" in docs["KI04"]["damage_origin"]

    assert "damage-profile PDF" in rec["extra"]["documentation_source"]
    assert rec["metrics"]["confusion_labels"] == [
        "Normal", "InnerRace", "OuterRace"]
    assert "one physical bearing per class" in rec["notes"]
    # Identical grouped protocol to wave 4.
    assert "GroupShuffleSplit" in rec["split"]["protocol"]
    assert rec["split"]["n_measurements_train"] == 192
    assert rec["split"]["n_measurements_test"] == 48
