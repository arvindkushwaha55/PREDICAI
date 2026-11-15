"""IMS Bearing Dataset adapter (NASA PCoE / University of Cincinnati).

Task C (anomaly detection / degradation monitoring) on run-to-failure
test-rig data. This wave integrates **Test 1 only**:

- 2,156 snapshot files, 2003-10-22 12:06 to 2003-11-25 23:39,
  nominally every 10 minutes (denser at the start of the run).
- Each file: 20,480 samples x 8 channels of accelerometer data at
  20 kHz (1 s of vibration per snapshot). Four bearings on one shaft,
  two accelerometers (x/y) per bearing:
  columns 1-2 = bearing 1, 3-4 = bearing 2, 5-6 = bearing 3,
  7-8 = bearing 4.
- Documented teardown ground truth for Test 1 (Readme Document for
  IMS Bearing Data, Lee et al.): **bearing 3 — inner-race defect;
  bearing 4 — roller-element defect**; bearings 1 and 2 had no
  reported defect. There are NO per-snapshot labels — the run is one
  continuous degradation trajectory per bearing, so honest evaluation
  is alarm TIMING (when does a detector fitted on early life first
  fire, and does it fire on the bearings that actually failed), not
  classification accuracy.

Provenance (verified 2026-11-15): the official NASA Prognostics Data
Repository page publishes the direct link
https://phm-datasets.s3.amazonaws.com/NASA/4.+Bearings.zip
(1,075,597,174 B, SHA-256
21001ac266c465f5d345ec42d7b508c6a6328487fd9d4d7774422dd5ea10ad83).
The archive is triple-nested: zip -> IMS.7z -> {1st,2nd,3rd}_test.rar.
Only 1st_test.rar was extracted (366,567,310 B) for this wave; Tests
2-3 remain in the local archive, unextracted. Licence: no per-dataset
licence is stated on the PCoE page (acknowledgement requested; use at
own risk) — see DATASET_REGISTRY.md.

The adapter emits ONE ROW PER SNAPSHOT: per-channel time-domain
features (RMS, std, excess kurtosis, crest factor, peak) for each of
the 8 channels, named bearing{b}_ch{c}_{feature}, plus the snapshot
index and timestamp parsed from the filename. Adapters never fit
anything (see base.py) — the anomaly detector lives in the
experiment layer.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kurtosis

from .base import DatasetAdapter

SAMPLE_RATE_HZ = 20_000
SNAPSHOT_POINTS = 20_480
N_BEARINGS = 4
CHANNELS_PER_BEARING = 2
N_CHANNELS = N_BEARINGS * CHANNELS_PER_BEARING

CHANNEL_FEATURES = ["rms", "std", "kurtosis", "crest_factor", "peak"]

GROUND_TRUTH_TEST1 = {
    1: None,
    2: None,
    3: "inner-race defect (found at teardown)",
    4: "roller-element defect (found at teardown)",
}


def channel_features(x: np.ndarray) -> dict:
    """Time-domain features of one channel of one snapshot."""
    x = np.asarray(x, dtype=np.float64)
    rms = float(np.sqrt(np.mean(x ** 2)))
    peak = float(np.max(np.abs(x)))
    return {
        "rms": rms,
        "std": float(np.std(x)),
        "kurtosis": float(kurtosis(x)),  # Fisher (excess) kurtosis
        "crest_factor": float(peak / rms) if rms > 0 else 0.0,
        "peak": peak,
    }


def parse_timestamp(filename: str) -> datetime:
    """IMS snapshot filenames are 'YYYY.MM.DD.HH.MM.SS' timestamps."""
    return datetime.strptime(filename, "%Y.%m.%d.%H.%M.%S")


def bearing_columns(bearing: int) -> list[str]:
    """Feature column names belonging to one bearing (1-based)."""
    return [f"bearing{bearing}_ch{c}_{f}"
            for c in range(1, CHANNELS_PER_BEARING + 1)
            for f in CHANNEL_FEATURES]


FEATURE_COLUMNS = [f"bearing{b}_ch{c}_{f}"
                   for b in range(1, N_BEARINGS + 1)
                   for c in range(1, CHANNELS_PER_BEARING + 1)
                   for f in CHANNEL_FEATURES]


class IMSAdapter(DatasetAdapter):
    name = "ims"

    def __init__(self, raw_dir: str | Path, test: str = "1st_test") -> None:
        super().__init__(raw_dir)
        if test != "1st_test":
            raise ValueError("only 1st_test is extracted in this wave")
        self.test = test

    def load(self) -> tuple[pd.DataFrame, dict]:
        test_dir = self.raw_dir / self.test
        files = sorted(test_dir.iterdir()) if test_dir.exists() else []
        files = [p for p in files if p.is_file()]
        if not files:
            raise FileNotFoundError(
                f"no IMS snapshot files under {test_dir} — extract "
                f"1st_test.rar from the official archive (see adapter "
                f"docstring for the verified provenance chain)")
        rows = []
        for i, path in enumerate(files):
            data = np.loadtxt(path)
            if data.shape != (SNAPSHOT_POINTS, N_CHANNELS):
                raise ValueError(
                    f"{path.name}: expected shape "
                    f"({SNAPSHOT_POINTS}, {N_CHANNELS}), got "
                    f"{data.shape}")
            row = {"snapshot_index": i,
                   "timestamp": parse_timestamp(path.name)}
            for ch in range(N_CHANNELS):
                bearing = ch // CHANNELS_PER_BEARING + 1
                chan = ch % CHANNELS_PER_BEARING + 1
                feats = channel_features(data[:, ch])
                for fname, value in feats.items():
                    row[f"bearing{bearing}_ch{chan}_{fname}"] = value
            rows.append(row)
        df = pd.DataFrame(rows)
        metadata = {
            "dataset": "ims",
            "source": "NASA Prognostics Data Repository (PCoE), entry "
                      "#4 'Bearings' — IMS bearing run-to-failure data "
                      "(Univ. of Cincinnati; Lee, Qiu, Yu, Lin & "
                      "Rexnord, 2007); official S3 link in "
                      "DATASET_REGISTRY.md",
            "licence": "not stated on the PCoE page (acknowledgement "
                       "requested; use at own risk)",
            "tasks": ["C"],
            "target": "none — no per-snapshot labels exist; Test 1 "
                      "teardown ground truth is per-bearing (see "
                      "ground_truth_test1)",
            "maintenance_history": "absent",
            "data_nature": "real run-to-failure test-rig recordings; "
                           "four bearings on one shaft run to failure "
                           "under constant radial load (6,000 lbs, "
                           "2,000 rpm)",
            "test": self.test,
            "n_snapshots": int(len(df)),
            "snapshot_points": SNAPSHOT_POINTS,
            "sample_rate_hz": SAMPLE_RATE_HZ,
            "channels": "8 = 4 bearings x 2 accelerometers (x/y); "
                        "columns 1-2 bearing 1, 3-4 bearing 2, "
                        "5-6 bearing 3, 7-8 bearing 4",
            "feature_columns": FEATURE_COLUMNS,
            "channel_features": CHANNEL_FEATURES,
            "ground_truth_test1": {
                f"bearing_{b}": v
                for b, v in GROUND_TRUTH_TEST1.items()},
            "evaluation_framing": "alarm timing only — no "
                                  "per-snapshot labels, so no "
                                  "precision/recall can be computed "
                                  "honestly",
        }
        return df, metadata
