"""PREDICAI platform — wave 5, IMS Bearing integration (Task C).

Kept in its own module because the data path is unlike every other
dataset in the platform: the official NASA PCoE archive is
triple-nested (zip -> 7z -> per-test rar) and Test 1 alone is 2,156
snapshot files. See platform_lib/adapters/ims.py for the verified
provenance chain and DATASET_REGISTRY.md for the licence status.

Experiment design (the only honest one for this dataset): there are
NO per-snapshot labels. One Isolation Forest per bearing is fitted on
the first 30% of the run (early life = nominal by construction of the
run-to-failure test), then scored over all 2,156 snapshots. The
threshold is the 95th percentile of the TRAIN anomaly scores. The
reported quantity is ALARM TIMING — the first snapshot from which the
score stays above threshold for >= 10 consecutive snapshots
(~100 minutes at the nominal 10-minute cadence) — compared against
the documented teardown findings for Test 1: bearing 3 inner-race
defect, bearing 4 roller-element defect, bearings 1-2 no defect
reported. No precision/recall/F1 is computed: inventing per-snapshot
labels to produce them would be fabrication.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

PLATFORM_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = PLATFORM_DIR.parent
ML_SRC = REPO_ROOT / "ml" / "src"
if str(ML_SRC) not in sys.path:
    sys.path.insert(0, str(ML_SRC))
if str(PLATFORM_DIR) not in sys.path:
    sys.path.insert(0, str(PLATFORM_DIR))

from platform_lib.benchmarking.run_benchmark import (  # noqa: E402
    RAW,
    SEED,
    base_record,
    save_record,
)

TRAIN_FRACTION = 0.30
THRESHOLD_QUANTILE = 0.95
SUSTAINED_SNAPSHOTS = 10


def _first_sustained(scores: np.ndarray, threshold: float,
                     start: int) -> int | None:
    """First index >= start from which `scores` stay above `threshold`
    for SUSTAINED_SNAPSHOTS consecutive snapshots (None if never)."""
    above = scores > threshold
    run = 0
    for i in range(start, len(scores)):
        run = run + 1 if above[i] else 0
        if run >= SUSTAINED_SNAPSHOTS:
            return i - SUSTAINED_SNAPSHOTS + 1
    return None


def run_ims() -> None:
    from sklearn.ensemble import IsolationForest

    from platform_lib.adapters.ims import (GROUND_TRUTH_TEST1,
                                           IMSAdapter, bearing_columns)

    print("=== IMS Bearing Test 1 — Task C — Isolation Forest "
          "(alarm timing) ===", flush=True)
    df, meta = IMSAdapter(RAW / "ims").load()
    n = len(df)
    n_train = int(TRAIN_FRACTION * n)
    print(f"  snapshots: {n}; nominal train window: first {n_train} "
          f"({df['timestamp'].iloc[0]} .. "
          f"{df['timestamp'].iloc[n_train - 1]})", flush=True)

    per_bearing = {}
    total_train_s, total_score_s = 0.0, 0.0
    for bearing in (1, 2, 3, 4):
        cols = bearing_columns(bearing)
        X = df[cols].to_numpy(dtype=np.float64)
        est = IsolationForest(n_estimators=200, contamination=0.01,
                              random_state=SEED, n_jobs=-1)
        t0 = time.perf_counter()
        est.fit(X[:n_train])
        total_train_s += time.perf_counter() - t0
        t0 = time.perf_counter()
        scores = -est.score_samples(X)  # higher = more anomalous
        total_score_s += time.perf_counter() - t0
        threshold = float(np.quantile(scores[:n_train],
                                      THRESHOLD_QUANTILE))
        alarm = _first_sustained(scores, threshold, start=n_train)
        last_decile = scores[int(0.9 * n):]
        entry = {
            "documented_defect": GROUND_TRUTH_TEST1[bearing],
            "threshold_train_q95": threshold,
            "train_exceedance_rate": float(
                np.mean(scores[:n_train] > threshold)),
            "first_sustained_alarm_index": alarm,
            "first_sustained_alarm_timestamp": (
                str(df["timestamp"].iloc[alarm])
                if alarm is not None else None),
            "fraction_of_run_before_alarm": (
                round(alarm / n, 4) if alarm is not None else None),
            "snapshots_of_warning_before_run_end": (
                int(n - alarm) if alarm is not None else 0),
            "last_decile_exceedance_rate": float(
                np.mean(last_decile > threshold)),
            "max_score_snapshot_index": int(np.argmax(scores)),
        }
        per_bearing[f"bearing_{bearing}"] = entry
        print(f"  bearing {bearing} "
              f"({GROUND_TRUTH_TEST1[bearing] or 'no defect reported'}"
              f"): first sustained alarm at "
              f"{alarm if alarm is not None else 'never'} / {n}",
              flush=True)

    rec = base_record(
        "ims", "C", "isolation_forest",
        {"n_estimators": 200, "contamination": 0.01,
         "random_state": SEED,
         "model_per_bearing": "one Isolation Forest per bearing on "
                              "that bearing's 10 channel features "
                              "(2 channels x rms/std/kurtosis/crest/"
                              "peak)",
         "threshold": f"{THRESHOLD_QUANTILE:.2f} quantile of the "
                      f"bearing's own train-window anomaly scores",
         "sustained_alarm_definition": f"score above threshold for "
                                       f">= {SUSTAINED_SNAPSHOTS} "
                                       f"consecutive snapshots"})
    rec["experiment_id"] = "ims__taskC__isolation_forest__s42"
    rec["split"] = {
        "protocol": "TEMPORAL split within one run-to-failure test: "
                    "each bearing's detector is fitted ONLY on the "
                    "first 30% of snapshots (early life, nominal by "
                    "construction) and scored over the full run. No "
                    "random splitting anywhere — snapshot order is "
                    "the run's chronology.",
        "n_snapshots": int(n),
        "n_train_snapshots_per_bearing": int(n_train),
        "train_window_timestamps": [str(df["timestamp"].iloc[0]),
                                    str(df["timestamp"].iloc[
                                        n_train - 1])],
        "full_run_timestamps": [str(df["timestamp"].iloc[0]),
                                str(df["timestamp"].iloc[-1])],
    }
    rec["preprocessing"] = {
        "features": meta["feature_columns"],
        "feature_note": "per-snapshot per-channel time-domain "
                        "features from IMSAdapter; no scaling (tree "
                        "model), no fitted statistics of any kind "
                        "beyond the detector itself and its "
                        "train-window threshold",
        "channels": meta["channels"],
    }
    rec["metrics"] = {
        "framing": "ALARM TIMING ONLY. IMS Test 1 has no "
                   "per-snapshot labels; the only ground truth is "
                   "the teardown finding (bearing 3 inner-race "
                   "defect, bearing 4 roller-element defect, bearings "
                   "1-2 none reported). Precision/recall/F1 are "
                   "therefore NOT computable and are not reported.",
        "alarm_timing": per_bearing,
    }
    rec["timing"] = {
        "train_s": round(total_train_s, 4),
        "infer_ms_per_row": round(total_score_s / (4 * n) * 1000.0, 5),
    }
    rec["notes"] = (
        "IMS Test 1 integrated in wave 5 after the user approved the "
        "1.08 GB official download that wave 3's size gate had "
        "blocked. Detectors see only their own bearing's channels. "
        "A sustained alarm on bearings 3/4 late in the run, with "
        "bearings 1/2 quiet or much later, is the success pattern "
        "this design can honestly show; whatever the measured "
        "pattern is, it is reported as measured in "
        "EXPERIMENTS_REPORT.md. Data is real run-to-failure rig data "
        "(not field data); licence not stated on the PCoE page.")
    rec["extra"] = {
        "archive_provenance": {
            "url": "https://phm-datasets.s3.amazonaws.com/NASA/"
                   "4.+Bearings.zip",
            "zip_bytes": 1075597174,
            "zip_sha256": "21001ac266c465f5d345ec42d7b508c6a6328"
                          "487fd9d4d7774422dd5ea10ad83",
            "nesting": "zip -> IMS.7z -> 1st_test.rar -> 2,156 "
                       "timestamped snapshot files",
            "extracted_scope": "Test 1 only (2,156 files); Tests 2-3 "
                               "remain unextracted in the local "
                               "archive",
        },
        "ground_truth_test1": meta["ground_truth_test1"],
    }
    b3 = per_bearing["bearing_3"]["first_sustained_alarm_index"]
    b4 = per_bearing["bearing_4"]["first_sustained_alarm_index"]
    save_record(rec, {"n_snapshots": int(n),
                      "bearing3_first_sustained_alarm": b3,
                      "bearing4_first_sustained_alarm": b4})


if __name__ == "__main__":
    run_ims()
