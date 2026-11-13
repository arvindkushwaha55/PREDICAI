"""Unit tests for the simulator's pure helpers (Part 10 / Phase E).

Only the pure functions are tested here — true-RUL arithmetic, window
slicing, machine-id format. The end-to-end replay (simulator -> Rust API
-> SQLite predictions_log) is proven by running it, not by a unit test.
"""
import sys
from pathlib import Path

import numpy as np

SIM_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SIM_DIR))
sys.path.insert(0, str(SIM_DIR.parent / "ml" / "src"))

from simulator import iter_windows, machine_id_for, true_rul_at  # noqa: E402


# --- true RUL --------------------------------------------------------------
def test_true_rul_at_last_cycle_equals_published_rul():
    # Engine 1: 31 observed cycles, RUL_FD001 = 112.
    assert true_rul_at(112, 31, 31) == 112.0


def test_true_rul_counts_back_cycle_by_cycle():
    assert true_rul_at(112, 31, 30) == 113.0
    assert true_rul_at(112, 31, 1) == 142.0
    # Engine 100: 198 observed cycles, RUL_FD001 = 20.
    assert true_rul_at(20, 198, 100) == 118.0


# --- window slicing ----------------------------------------------------------
def _rows(n: int) -> np.ndarray:
    # Distinct, ordered values so window content/order is checkable:
    # row i (0-based) is [i, 1000 + i].
    return np.column_stack([np.arange(n, dtype=float), 1000 + np.arange(n, dtype=float)])


def test_no_window_before_30_cycles():
    assert list(iter_windows(_rows(29))) == []
    wins = list(iter_windows(_rows(30)))
    assert len(wins) == 1
    assert wins[0][0] == 30  # first window ends at cycle 30


def test_window_count_and_end_cycles():
    wins = list(iter_windows(_rows(35)))
    assert len(wins) == 6  # cycles 30..35
    assert [end for end, _ in wins] == [30, 31, 32, 33, 34, 35]


def test_window_contents_are_last_30_raw_rows_in_order():
    rows = _rows(35)
    wins = dict(iter_windows(rows))
    np.testing.assert_array_equal(wins[30], rows[0:30])
    np.testing.assert_array_equal(wins[35], rows[5:35])
    # Last row of the final window is the last recorded row, unchanged (raw).
    np.testing.assert_array_equal(wins[35][-1], rows[-1])


# --- machine id ---------------------------------------------------------------
def test_machine_id_format():
    assert machine_id_for(1) == "ENGINE-001"
    assert machine_id_for(49) == "ENGINE-049"
    assert machine_id_for(100) == "ENGINE-100"
