"""Dataset adapter interface (ARCHITECTURE.md §2).

An adapter's ONLY job is ingestion: turn a dataset's raw files into a
canonical pandas DataFrame plus a metadata dict. Adapters never impute,
scale, encode, or split — fitted preprocessing belongs to the experiment
layer so that every fitted statistic is computed on training data only
and serialised into the experiment record.

Metadata contract (required keys):
    dataset              short id, e.g. "ai4i2020"
    source               human-readable source + URL
    tasks                list of platform tasks served: "A" / "B" / "C"
    target               target column name (tabular) or description
    maintenance_history  "absent" unless the dataset truly contains service
                         logs — the platform never invents them
Adapters may add dataset-specific keys (label maps, leakage columns, ...)
and extra loader methods (e.g. CMapssAdapter.load_test).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd


class DatasetAdapter(ABC):
    name: str = "base"

    def __init__(self, raw_dir: str | Path) -> None:
        self.raw_dir = Path(raw_dir)

    @abstractmethod
    def load(self) -> tuple[pd.DataFrame, dict]:
        """Return (dataframe, metadata) for the dataset's primary frame."""
        raise NotImplementedError
