"""NASA C-MAPSS adapter (Task B) — subset-parameterised.

Wraps the existing, tested loader in `ml/src/predicai_ml/data.py` instead
of duplicating the 26-column schema. `load()` returns the TRAIN frame;
`load_test()` returns (test frame, true RUL array) for the official
last-window-per-engine protocol.

Subset facts (verified in DATASET_RESEARCH.md §1):
    FD001  1 operating condition, 1 fault mode   (HPC)
    FD002  6 operating conditions, 1 fault mode  (HPC)
    FD003  1 operating condition, 2 fault modes  (HPC + fan)
    FD004  6 operating conditions, 2 fault modes (HPC + fan)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Reuse the existing FD001 system's loader — never re-implement the schema.
_ML_SRC = Path(__file__).resolve().parents[3] / "ml" / "src"
if str(_ML_SRC) not in sys.path:
    sys.path.insert(0, str(_ML_SRC))

from predicai_ml.data import load_cmapss_file, load_rul_file  # noqa: E402

from .base import DatasetAdapter  # noqa: E402

SUBSETS = {
    "FD001": {"operating_conditions": 1, "fault_modes": 1},
    "FD002": {"operating_conditions": 6, "fault_modes": 1},
    "FD003": {"operating_conditions": 1, "fault_modes": 2},
    "FD004": {"operating_conditions": 6, "fault_modes": 2},
}


class CMapssAdapter(DatasetAdapter):
    name = "cmapss"

    def __init__(self, raw_dir: str | Path, subset: str = "FD003") -> None:
        super().__init__(raw_dir)
        if subset not in SUBSETS:
            raise ValueError(f"unknown C-MAPSS subset {subset!r}")
        self.subset = subset

    def load(self) -> tuple[pd.DataFrame, dict]:
        train = load_cmapss_file(self.raw_dir / f"train_{self.subset}.txt")
        info = SUBSETS[self.subset]
        metadata = {
            "dataset": f"cmapss_{self.subset.lower()}",
            "source": "NASA Ames PCoE, C-MAPSS Turbofan Engine Degradation "
                      "Simulation (Saxena & Goebel 2008), PHM Society mirror",
            "tasks": ["B"],
            "target": "rul (cycles, derived: life - cycle for train engines; "
                      "RUL file + observed life for truncated test engines)",
            "operating_conditions": info["operating_conditions"],
            "fault_modes": info["fault_modes"],
            "maintenance_history": "absent",
            "data_nature": "simulated (C-MAPSS tool), not real engines",
            "n_train_rows": int(len(train)),
            "n_train_engines": int(train["unit"].nunique()),
        }
        return train, metadata

    def load_test(self) -> tuple[pd.DataFrame, np.ndarray]:
        test = load_cmapss_file(self.raw_dir / f"test_{self.subset}.txt")
        rul = load_rul_file(self.raw_dir / f"RUL_{self.subset}.txt")
        return test, rul
