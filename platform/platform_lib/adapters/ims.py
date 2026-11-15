"""IMS Bearing Dataset adapter (NASA PCoE / University of Cincinnati).

Task C (anomaly detection / degradation monitoring) on run-to-failure
test-rig data. Wave 5 integrated Test 1; wave 9 (2026-11-15) added
Tests 2 and 3 from the same local archive, so all three tests are now
supported:

- Test 1: 2,156 snapshot files, 2003-10-22 12:06 to 2003-11-25 23:39,
  nominally every 10 minutes (denser at the start of the run).
  Each file: 20,480 samples x 8 channels at 20 kHz (1 s of vibration
  per snapshot). Four bearings on one shaft, TWO accelerometers (x/y)
  per bearing: columns 1-2 = bearing 1, 3-4 = bearing 2,
  5-6 = bearing 3, 7-8 = bearing 4.
- Test 2: 984 snapshot files, 2004-02-12 10:32 to 2004-02-19 06:22,
  every 10 minutes. 20,480 samples x 4 channels — ONE accelerometer
  per bearing (column k = bearing k), per the Readme Document.
- Test 3: 4,448 snapshot files, 2004-03-04 09:27 to 2004-04-04 19:01,
  every 10 minutes. 20,480 samples x 4 channels, same arrangement
  as Test 2. ARCHIVE QUIRK (found in wave 9): 3rd_test.rar actually
  contains 6,324 files — the first 4,448 are exactly the
  Readme-documented test-to-failure experiment (ending
  2004-04-04 19:01:57, to the second); the remaining 1,876 files
  (2004-04-04 19:11 to 2004-04-18) fall AFTER the documented
  experiment and its teardown finding, so they are extracted to
  `3rd_test_beyond_documented_period/` and are NOT part of the
  test this adapter loads. The Readme does not state what those
  later files are; no purpose is assumed for them here.
- Documented teardown ground truth (Readme Document for IMS Bearing
  Data, Lee et al.; verified against the copy inside the official
  archive in wave 9): **Test 1 — bearing 3 inner-race defect,
  bearing 4 roller-element defect; Test 2 — bearing 1 outer-race
  failure; Test 3 — bearing 3 outer-race failure**; all other
  bearings had no reported defect. There are NO per-snapshot
  labels — each run is one continuous degradation trajectory per
  bearing, so honest evaluation is alarm TIMING (when does a
  detector fitted on early life first fire, and does it fire on
  the bearings that actually failed), not classification accuracy.

Provenance (verified 2026-11-15; zip SHA-256 re-verified in wave 9):
the official NASA Prognostics Data Repository page publishes the
direct link https://phm-datasets.s3.amazonaws.com/NASA/4.+Bearings.zip
(1,075,597,174 B, SHA-256
21001ac266c465f5d345ec42d7b508c6a6328487fd9d4d7774422dd5ea10ad83).
The archive is triple-nested: zip -> IMS.7z -> {1st,2nd,3rd}_test.rar
(366,567,310 B / 85,581,092 B / 609,047,134 B). Licence: no
per-dataset licence is stated on the PCoE page (acknowledgement
requested; use at own risk) — see DATASET_REGISTRY.md.

The adapter emits ONE ROW PER SNAPSHOT: per-channel time-domain
features (RMS, std, excess kurtosis, crest factor, peak) for each
channel (8 in Test 1, 4 in Tests 2-3), named
bearing{b}_ch{c}_{feature}, plus the snapshot
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
CHANNELS_PER_BEARING = 2  # Test 1; Tests 2-3 have 1 (see TEST_CHANNELS)
N_CHANNELS = N_BEARINGS * CHANNELS_PER_BEARING

CHANNEL_FEATURES = ["rms", "std", "kurtosis", "crest_factor", "peak"]

# Total accelerometer channels per test (Readme Document, verified
# wave 9): Test 1 recorded two accelerometers (x/y) per bearing;
# Tests 2-3 recorded one per bearing.
TEST_CHANNELS = {"1st_test": 8, "2nd_test": 4, "3rd_test": 4}

GROUND_TRUTH_TEST1 = {
    1: None,
    2: None,
    3: "inner-race defect (found at teardown)",
    4: "roller-element defect (found at teardown)",
}

GROUND_TRUTH_TEST2 = {
    1: "outer-race failure (found at teardown)",
    2: None,
    3: None,
    4: None,
}

GROUND_TRUTH_TEST3 = {
    1: None,
    2: None,
    3: "outer-race failure (found at teardown)",
    4: None,
}

GROUND_TRUTH = {
    "1st_test": GROUND_TRUTH_TEST1,
    "2nd_test": GROUND_TRUTH_TEST2,
    "3rd_test": GROUND_TRUTH_TEST3,
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


def bearing_columns(bearing: int,
                      channels_per_bearing: int = CHANNELS_PER_BEARING
                      ) -> list[str]:
    """Feature column names belonging to one bearing (1-based)."""
    return [f"bearing{bearing}_ch{c}_{f}"
            for c in range(1, channels_per_bearing + 1)
            for f in CHANNEL_FEATURES]


def feature_columns_for(channels_per_bearing: int) -> list[str]:
    """All feature column names for a test with this many channels
    per bearing (2 for Test 1, 1 for Tests 2-3)."""
    return [f"bearing{b}_ch{c}_{f}"
            for b in range(1, N_BEARINGS + 1)
            for c in range(1, channels_per_bearing + 1)
            for f in CHANNEL_FEATURES]


FEATURE_COLUMNS = feature_columns_for(CHANNELS_PER_BEARING)


class IMSAdapter(DatasetAdapter):
    name = "ims"

    def __init__(self, raw_dir: str | Path, test: str = "1st_test") -> None:
        super().__init__(raw_dir)
        if test not in TEST_CHANNELS:
            raise ValueError(
                f"unknown IMS test {test!r}; expected one of "
                f"{sorted(TEST_CHANNELS)}")
        self.test = test
        self.n_channels = TEST_CHANNELS[test]
        self.channels_per_bearing = self.n_channels // N_BEARINGS

    def load(self) -> tuple[pd.DataFrame, dict]:
        test_dir = self.raw_dir / self.test
        files = sorted(test_dir.iterdir()) if test_dir.exists() else []
        files = [p for p in files if p.is_file()]
        if not files:
            raise FileNotFoundError(
                f"no IMS snapshot files under {test_dir} — extract "
                f"{self.test}.rar from the official archive (see "
                f"adapter docstring for the verified provenance "
                f"chain)")
        cpb = self.channels_per_bearing
        rows = []
        for i, path in enumerate(files):
            data = np.loadtxt(path)
            if data.shape != (SNAPSHOT_POINTS, self.n_channels):
                raise ValueError(
                    f"{path.name}: expected shape "
                    f"({SNAPSHOT_POINTS}, {self.n_channels}), got "
                    f"{data.shape}")
            row = {"snapshot_index": i,
                   "timestamp": parse_timestamp(path.name)}
            for ch in range(self.n_channels):
                bearing = ch // cpb + 1
                chan = ch % cpb + 1
                feats = channel_features(data[:, ch])
                for fname, value in feats.items():
                    row[f"bearing{bearing}_ch{chan}_{fname}"] = value
            rows.append(row)
        df = pd.DataFrame(rows)
        gt = GROUND_TRUTH[self.test]
        gt_key = {"1st_test": "ground_truth_test1",
                  "2nd_test": "ground_truth_test2",
                  "3rd_test": "ground_truth_test3"}[self.test]
        if cpb == 2:
            channels_desc = ("8 = 4 bearings x 2 accelerometers (x/y); "
                             "columns 1-2 bearing 1, 3-4 bearing 2, "
                             "5-6 bearing 3, 7-8 bearing 4")
        else:
            channels_desc = ("4 = 4 bearings x 1 accelerometer; "
                             "column k = bearing k")
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
            "target": "none — no per-snapshot labels exist; teardown "
                      "ground truth is per-bearing (see "
                      f"{gt_key})",
            "maintenance_history": "absent",
            "data_nature": "real run-to-failure test-rig recordings; "
                           "four bearings on one shaft run to failure "
                           "under constant radial load (6,000 lbs, "
                           "2,000 rpm)",
            "test": self.test,
            "n_snapshots": int(len(df)),
            "snapshot_points": SNAPSHOT_POINTS,
            "sample_rate_hz": SAMPLE_RATE_HZ,
            "channels": channels_desc,
            "channels_per_bearing": cpb,
            "feature_columns": feature_columns_for(cpb),
            "channel_features": CHANNEL_FEATURES,
            gt_key: {f"bearing_{b}": v for b, v in gt.items()},
            "evaluation_framing": "alarm timing only — no "
                                  "per-snapshot labels, so no "
                                  "classification scores can be "
                                  "computed honestly",
        }
        return df, metadata
