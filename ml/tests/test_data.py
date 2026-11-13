"""PREDICAI — dataset loader tests (Phase A1 gate)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from predicai_ml.data import COLUMNS, load_fd001  # noqa: E402

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"


def test_fd001_shapes_and_counts():
    train, test, rul = load_fd001(RAW)
    assert list(train.columns) == COLUMNS and len(COLUMNS) == 26
    assert train.shape == (20631, 26)
    assert test.shape == (13096, 26)
    assert train["unit"].nunique() == 100
    assert test["unit"].nunique() == 100
    assert len(rul) == 100


def test_no_missing_values():
    train, test, _ = load_fd001(RAW)
    assert int(train.isna().sum().sum()) == 0
    assert int(test.isna().sum().sum()) == 0


def test_cycles_start_at_1_and_increase_per_engine():
    train, _, _ = load_fd001(RAW)
    for _, grp in train.groupby("unit"):
        cycles = grp["cycle"].tolist()
        assert cycles[0] == 1
        assert cycles == sorted(cycles)
        assert len(set(cycles)) == len(cycles)
