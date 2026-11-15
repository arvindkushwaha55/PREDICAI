"""UCI AI4I 2020 Predictive Maintenance adapter (Task A anchor dataset).

Raw layout: a single CSV `ai4i2020.csv` (10,000 rows x 14 cols).
Quirks handled here (and only here):
- The CSV carries a UTF-8 BOM, so it is read with encoding="utf-8-sig".
- The five failure-mode flags (TWF/HDF/PWF/OSF/RNF) *define* the target by
  construction; using them as inputs would be target leakage. They are
  declared in metadata["leakage_columns"] and the benchmark layer excludes
  them from features.
- UID / Product ID are identifiers, declared in metadata["id_columns"].
The data is synthetic (rule-generated failures) — metadata says so, and
every report built on it must repeat that caveat.
"""
from __future__ import annotations

import pandas as pd

from .base import DatasetAdapter

NUMERIC_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
FAILURE_MODE_COLUMNS = ["TWF", "HDF", "PWF", "OSF", "RNF"]


class AI4IAdapter(DatasetAdapter):
    name = "ai4i2020"

    def load(self) -> tuple[pd.DataFrame, dict]:
        path = self.raw_dir / "ai4i2020.csv"
        df = pd.read_csv(path, encoding="utf-8-sig")
        metadata = {
            "dataset": "ai4i2020",
            "source": "UCI ML Repository, AI4I 2020 Predictive Maintenance "
                      "(doi:10.24432/C5HS5C), CC BY 4.0",
            "tasks": ["A", "C"],
            "target": "Machine failure",
            "target_semantics": "1 = machine failure (any of 5 rule-generated modes)",
            "numeric_features": NUMERIC_FEATURES,
            "categorical_features": ["Type"],
            "id_columns": ["UDI", "Product ID"],
            "leakage_columns": FAILURE_MODE_COLUMNS,
            "maintenance_history": "absent",
            "data_nature": "synthetic (rule-generated failures on a milling "
                           "process model) — scores must not be read as "
                           "real-world machine performance",
            "n_rows": int(len(df)),
            "failure_rate": float(df["Machine failure"].mean()),
        }
        return df, metadata
