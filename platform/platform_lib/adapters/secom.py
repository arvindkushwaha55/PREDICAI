"""UCI SECOM Manufacturing adapter (Tasks A + C).

Raw layout: two whitespace-delimited files.
- `secom.data`        1,567 rows x 590 anonymised sensor columns, the
                      literal string "NaN" marks missing values.
- `secom_labels.data` 2 columns: label and timestamp. Label semantics in
                      the raw file: -1 = pass, +1 = fail. The adapter maps
                      them to the platform convention fail = 1, pass = 0
                      and exposes the result as the `target` column.
Features are anonymised (sensor_000 ... sensor_589 here as f0..f589):
explanations may only ever reference these indices, never physical names.
"""
from __future__ import annotations

import pandas as pd

from .base import DatasetAdapter


class SECOMAdapter(DatasetAdapter):
    name = "secom"

    def load(self) -> tuple[pd.DataFrame, dict]:
        data = pd.read_csv(
            self.raw_dir / "secom.data", sep=r"\s+", header=None, na_values="NaN"
        )
        data.columns = [f"f{i}" for i in range(data.shape[1])]
        labels = pd.read_csv(
            self.raw_dir / "secom_labels.data", sep=r"\s+", header=None,
            usecols=[0], names=["raw_label"],
        )
        if len(labels) != len(data):
            raise ValueError(
                f"SECOM label/data row mismatch: {len(labels)} vs {len(data)}"
            )
        # Raw: -1 = pass, +1 = fail  ->  platform: fail = 1, pass = 0
        data["target"] = (labels["raw_label"] == 1).astype(int)
        metadata = {
            "dataset": "secom",
            "source": "UCI ML Repository, SECOM (doi:10.24432/C54305), CC BY 4.0",
            "tasks": ["A", "C"],
            "target": "target",
            "target_semantics": "1 = fail (raw label +1), 0 = pass (raw label -1)",
            "raw_label_mapping": {"-1": "pass -> 0", "1": "fail -> 1"},
            "feature_columns": [c for c in data.columns if c != "target"],
            "features_anonymised": True,
            "maintenance_history": "absent",
            "data_nature": "real semiconductor process data (2008), anonymised",
            "n_rows": int(len(data)),
            "n_features_raw": int(data.shape[1] - 1),
            "fail_rate": float(data["target"].mean()),
            "missing_fraction_overall": float(data.drop(columns=["target"]).isna().mean().mean()),
        }
        return data, metadata
