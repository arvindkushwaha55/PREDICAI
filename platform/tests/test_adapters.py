"""Adapter tests — grounded in the verified dataset facts from
DATASET_RESEARCH.md (AI4I: 10,000 rows, 339 failures; SECOM: 1,567 rows,
104 fails / 1,463 pass; FD003: 24,720 train / 16,596 test rows, 100 engines).
"""
import pytest

from conftest import RAW
from platform_lib.adapters.ai4i import AI4IAdapter
from platform_lib.adapters.cmapss import CMapssAdapter
from platform_lib.adapters.secom import SECOMAdapter


@pytest.fixture(scope="module")
def ai4i():
    return AI4IAdapter(RAW / "ai4i").load()


def test_ai4i_shape_and_columns(ai4i):
    df, meta = ai4i
    assert df.shape == (10000, 14)
    for col in ("UDI", "Product ID", "Type", "Machine failure",
                "Torque [Nm]", "Tool wear [min]"):
        assert col in df.columns
    assert meta["target"] == "Machine failure"
    assert meta["maintenance_history"] == "absent"


def test_ai4i_failure_count_and_leakage_declaration(ai4i):
    df, meta = ai4i
    assert int(df["Machine failure"].sum()) == 339
    # failure-mode flags define the target -> must be declared as leakage
    assert meta["leakage_columns"] == ["TWF", "HDF", "PWF", "OSF", "RNF"]
    for col in meta["leakage_columns"]:
        assert col not in meta["numeric_features"]


@pytest.fixture(scope="module")
def secom():
    return SECOMAdapter(RAW / "secom").load()


def test_secom_shape(secom):
    df, meta = secom
    assert df.shape == (1567, 591)  # 590 features + mapped target
    assert meta["n_features_raw"] == 590


def test_secom_label_mapping(secom):
    df, meta = secom
    # raw -1 (pass) -> 0, raw +1 (fail) -> 1
    assert set(df["target"].unique()) == {0, 1}
    assert int((df["target"] == 1).sum()) == 104
    assert int((df["target"] == 0).sum()) == 1463
    assert meta["raw_label_mapping"]["-1"] == "pass -> 0"
    assert meta["raw_label_mapping"]["1"] == "fail -> 1"


@pytest.fixture(scope="module")
def cmapss_fd003():
    adapter = CMapssAdapter(RAW / "cmapss" / "extracted", subset="FD003")
    train, meta = adapter.load()
    test, rul = adapter.load_test()
    return train, test, rul, meta


def test_cmapss_fd003_shapes(cmapss_fd003):
    train, test, rul, meta = cmapss_fd003
    assert train.shape == (24720, 26)
    assert test.shape == (16596, 26)
    assert len(rul) == 100
    assert train["unit"].nunique() == 100
    # FD003: single operating condition, two fault modes (research-verified)
    assert meta["operating_conditions"] == 1
    assert meta["fault_modes"] == 2
