"""Wave-3 tests (advanced expansion, Waves 1-3, 2026-11-15).

Covered:
- NASA Battery adapter (Task B): per-discharge-cycle table built from the
  official .mat files — discharge counts per battery, the capacity-fade
  sanity range, EOL (1.4 Ah) crossing indices, and RUL-label arithmetic.
  Facts asserted below come from the downloaded files themselves
  (B0005: 168 discharges, first capacity 1.8565 Ah, EOL crossing at
  discharge 125; B0007 never crosses 1.4 Ah).
- Paderborn adapter (Task A): vibration-channel extraction from a real
  measurement file (256,001 samples @ 64 kHz), condition-code decoding,
  and a full adapter load over the 3 integrated bearings (80
  measurements each, 30,022 windows total).
- Registry presence: the central DATASET_REGISTRY.md must exist and
  state the blocked status of MFPT and IMS (documentation regression
  guard, not a data test).
All data tests skip cleanly when the (gitignored) raw files are absent.
"""
import numpy as np
import pytest

from conftest import PLATFORM_DIR, RAW

BATTERY_DIR = RAW / "nasa_battery"
PADERBORN_DIR = RAW / "paderborn"


def _battery_file(battery: str):
    matches = sorted(BATTERY_DIR.rglob(f"{battery}.mat"))
    return matches[0] if matches else None


pytestmark_battery = pytest.mark.skipif(
    _battery_file("B0005") is None,
    reason="NASA Battery official .mat files not downloaded")
pytestmark_paderborn = pytest.mark.skipif(
    not (PADERBORN_DIR / "K001").exists(),
    reason="Paderborn extracted .mat files not present")


# ---------------------------------------------------------------- battery

@pytestmark_battery
def test_battery_single_file_discharge_cycles():
    from platform_lib.adapters.nasa_battery import load_discharge_cycles

    rows = load_discharge_cycles(_battery_file("B0005"), "B0005")
    assert len(rows) == 168  # discharge operations in B0005 (of 616 total)
    assert [r["cycle_index"] for r in rows] == list(range(1, 169))
    caps = [r["capacity_ah"] for r in rows]
    assert caps[0] == pytest.approx(1.8565, abs=0.01)
    assert caps[-1] == pytest.approx(1.3251, abs=0.01)
    assert all(0.5 < c < 2.2 for c in caps)
    assert all(np.isfinite(r["duration_s"]) and r["duration_s"] > 0
               for r in rows)


@pytestmark_battery
def test_battery_adapter_rul_labels():
    from platform_lib.adapters.nasa_battery import NASABatteryAdapter

    df, meta = NASABatteryAdapter(BATTERY_DIR).load()
    assert meta["dataset"] == "nasa_battery"
    assert meta["tasks"] == ["B"]
    assert meta["maintenance_history"] == "absent"
    assert len(df) == 636  # 168 + 168 + 168 + 132 discharge cycles
    assert not df.isna().any().any()

    counts = df.groupby("battery")["cycle_index"].max().to_dict()
    assert counts == {"B0005": 168, "B0006": 168, "B0007": 168,
                      "B0018": 132}
    eol = {b["battery"]: b["eol_cycle_index"] for b in meta["batteries"]}
    assert eol == {"B0005": 125, "B0006": 109, "B0007": None,
                   "B0018": 97}

    for _, g in df.groupby("battery"):
        n = int(g["cycle_index"].max())
        # cycles-to-end is exact arithmetic on the discharge index
        assert (g["rul_cycles_to_end"]
                == n - g["cycle_index"]).all()
        assert (g["rul_cycles_eol"] >= 0).all()
    b5 = df[df["battery"] == "B0005"].set_index("cycle_index")
    assert b5.loc[1, "rul_cycles_eol"] == 124   # EOL crossing at 125
    assert b5.loc[125, "rul_cycles_eol"] == 0
    assert b5.loc[125, "capacity_ah"] < 1.4     # the crossing cycle itself
    # B0007 never crosses 1.4 Ah -> documented fallback = cycles to end
    b7 = df[df["battery"] == "B0007"]
    assert (b7["capacity_ah"] >= 1.4).all()
    assert (b7["rul_cycles_eol"] == b7["rul_cycles_to_end"]).all()


# -------------------------------------------------------------- paderborn

@pytestmark_paderborn
def test_paderborn_vibration_channel():
    from platform_lib.adapters.paderborn import load_vibration

    path = PADERBORN_DIR / "K001" / "N15_M07_F10_K001_1.mat"
    series = load_vibration(path)
    assert len(series) == 256001  # 4 s @ 64 kHz (+1 sample)
    assert np.isfinite(series).all()


def test_paderborn_condition_decoding():
    from platform_lib.adapters.paderborn import parse_condition

    assert parse_condition("N15_M07_F10") == {
        "condition": "N15_M07_F10", "speed_rpm_nominal": 1500,
        "torque_nm_nominal": 0.7, "radial_force_n_nominal": 1000}
    assert parse_condition("N09_M01_F04")["speed_rpm_nominal"] == 900
    assert parse_condition("N09_M01_F04")["torque_nm_nominal"] == 0.1
    assert parse_condition("N09_M01_F04")["radial_force_n_nominal"] == 400


@pytestmark_paderborn
def test_paderborn_adapter_load():
    from platform_lib.adapters.paderborn import PaderbornAdapter

    df, meta = PaderbornAdapter(PADERBORN_DIR).load()
    assert meta["dataset"] == "paderborn"
    assert meta["licence"].startswith("CC BY-NC 4.0")
    assert meta["maintenance_history"] == "absent"
    assert len(df) == 30022
    assert not df.isna().any().any()
    counts = df["label"].value_counts().to_dict()
    assert counts == {"Normal": 10006, "OuterRace": 10015,
                      "InnerRace": 10001}
    assert set(df["condition"]) == {"N15_M07_F10", "N15_M01_F10",
                                    "N09_M07_F10", "N15_M07_F04"}
    per_bearing = {b["bearing"]: b for b in meta["bearings"]}
    assert all(b["n_measurements"] == 80 for b in per_bearing.values())
    assert per_bearing["K001"]["label"] == "Normal"
    assert per_bearing["KA01"]["damage_origin"].startswith("artificial")
    assert per_bearing["KI01"]["label"] == "InnerRace"
    for col in meta["feature_columns"]:
        assert col in df.columns


# --------------------------------------------------------------- registry

def test_registry_documents_blocked_datasets():
    registry = (PLATFORM_DIR / "DATASET_REGISTRY.md").read_text()
    assert "MFPT" in registry and "IMS" in registry
    # Both must be presented with a blocked status, never as integrated.
    assert registry.count("blocked") >= 3  # Bosch + MFPT + IMS (+ vocab)
    assert "no verified public" in registry  # automotive conclusion
