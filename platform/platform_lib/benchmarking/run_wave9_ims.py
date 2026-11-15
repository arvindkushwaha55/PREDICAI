"""PREDICAI platform — wave 9, IMS Bearing Tests 2 and 3 (Task C).

Wave 5 integrated IMS Test 1 (see run_wave5_ims.py for the design
rationale). Wave 9 completes the dataset: Tests 2 and 3 are extracted
from the same local official archive (zip SHA-256 re-verified against
the value recorded in wave 5) and run through the IDENTICAL
per-bearing Isolation Forest protocol — the protocol constants and
the sustained-alarm helper are IMPORTED from run_wave5_ims, not
re-typed, so the three tests are comparable by construction:

- one Isolation Forest per bearing (200 trees, contamination 0.01,
  seed 42), fitted ONLY on the first 30% of snapshots;
- threshold = 95th percentile of the bearing's own train-window
  anomaly scores;
- alarm = first snapshot (search starts after the train window) from
  which the score stays above threshold for >= 10 consecutive
  snapshots;
- reported quantities are ALARM TIMING ONLY.

Documented teardown ground truth (Readme Document for IMS Bearing
Data, verified against the copy inside the official archive):
Test 2 — bearing 1 outer-race failure; Test 3 — bearing 3 outer-race
failure; all other bearings in those tests had no reported defect.

ONE documented difference from Test 1, a property of the data rather
than of the protocol: Tests 2-3 recorded ONE accelerometer per
bearing (4 channels total), where Test 1 recorded two (8 channels).
Each bearing's detector therefore sees 5 channel features instead of
10. This is stated in every record; nothing is padded or imputed to
make the feature counts match.

Usage (from the platform/ directory):
    ../.venv/bin/python -m platform_lib.benchmarking.run_wave9_ims \
        --set test2 | test3 | all
"""
from __future__ import annotations

import argparse
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
from platform_lib.benchmarking.run_wave5_ims import (  # noqa: E402
    SUSTAINED_SNAPSHOTS,
    THRESHOLD_QUANTILE,
    TRAIN_FRACTION,
    _first_sustained,
)

TESTS = {
    "test2": ("2nd_test", "ims_test2", 1),
    "test3": ("3rd_test", "ims_test3", 3),
}


def run_ims_test(set_name: str) -> None:
    from sklearn.ensemble import IsolationForest

    from platform_lib.adapters.ims import (GROUND_TRUTH, IMSAdapter,
                                           bearing_columns)

    test_dir_name, dataset_name, failed_bearing = TESTS[set_name]
    ground_truth = GROUND_TRUTH[test_dir_name]
    print(f"=== IMS Bearing {test_dir_name} — Task C — Isolation "
          f"Forest (alarm timing) ===", flush=True)
    adapter = IMSAdapter(RAW / "ims", test=test_dir_name)
    df, meta = adapter.load()
    cpb = adapter.channels_per_bearing
    n = len(df)
    n_train = int(TRAIN_FRACTION * n)
    print(f"  snapshots: {n}; channels: {adapter.n_channels} "
          f"({cpb} per bearing); nominal train window: first "
          f"{n_train} ({df['timestamp'].iloc[0]} .. "
          f"{df['timestamp'].iloc[n_train - 1]})", flush=True)

    per_bearing = {}
    total_train_s, total_score_s = 0.0, 0.0
    for bearing in (1, 2, 3, 4):
        cols = bearing_columns(bearing, channels_per_bearing=cpb)
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
            "documented_defect": ground_truth[bearing],
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
              f"({ground_truth[bearing] or 'no defect reported'}): "
              f"first sustained alarm at "
              f"{alarm if alarm is not None else 'never'} / {n}",
              flush=True)

    failed_key = f"bearing_{failed_bearing}"
    rec = base_record(
        dataset_name, "C", "isolation_forest",
        {"n_estimators": 200, "contamination": 0.01,
         "random_state": SEED,
         "model_per_bearing": "one Isolation Forest per bearing on "
                              "that bearing's channel features "
                              f"({cpb} channel(s) x rms/std/kurtosis/"
                              f"crest/peak = {5 * cpb} features)",
         "threshold": f"{THRESHOLD_QUANTILE:.2f} quantile of the "
                      f"bearing's own train-window anomaly scores",
         "sustained_alarm_definition": f"score above threshold for "
                                       f">= {SUSTAINED_SNAPSHOTS} "
                                       f"consecutive snapshots",
         "protocol_identity": "protocol constants and the "
                              "sustained-alarm helper are imported "
                              "from run_wave5_ims (the Test 1 runner) "
                              "— identical protocol by construction"})
    rec["experiment_id"] = f"{dataset_name}__taskC__isolation_forest__s42"
    rec["split"] = {
        "protocol": "TEMPORAL split within one run-to-failure test: "
                    "each bearing's detector is fitted ONLY on the "
                    "first 30% of snapshots (early life, nominal by "
                    "construction) and scored over the full run. No "
                    "random splitting anywhere — snapshot order is "
                    "the run's chronology. Identical to the wave-5 "
                    "Test 1 protocol.",
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
        "channel_difference_vs_test1": (
            "Tests 2-3 recorded ONE accelerometer per bearing "
            "(4 channels total); Test 1 recorded two per bearing "
            "(8 channels). Each bearing's detector here therefore "
            "uses 5 features instead of 10 — a property of the "
            "recorded data, stated rather than corrected."),
    }
    rec["metrics"] = {
        "framing": f"ALARM TIMING ONLY. IMS {test_dir_name} has no "
                   f"per-snapshot labels; the only ground truth is "
                   f"the teardown finding (bearing {failed_bearing} "
                   f"outer-race failure; the other bearings none "
                   f"reported). Classification scores are therefore "
                   f"NOT computable and are not reported.",
        "alarm_timing": per_bearing,
    }
    rec["timing"] = {
        "train_s": round(total_train_s, 4),
        "infer_ms_per_row": round(total_score_s / (4 * n) * 1000.0, 5),
    }
    rec["notes"] = (
        f"IMS {test_dir_name} integrated in wave 9, completing the "
        f"IMS dataset (Tests 1-3) under the identical wave-5 "
        f"protocol. Detectors see only their own bearing's channels. "
        f"A sustained alarm on bearing {failed_bearing} (the "
        f"documented failure) late in the run, with the other "
        f"bearings quiet or much later, is the success pattern this "
        f"design can honestly show; whatever the measured pattern "
        f"is, it is reported as measured in EXPERIMENTS_REPORT.md. "
        f"Data is real run-to-failure rig data (not field data); "
        f"licence not stated on the PCoE page.")
    rec["extra"] = {
        "archive_provenance": {
            "url": "https://phm-datasets.s3.amazonaws.com/NASA/"
                   "4.+Bearings.zip",
            "zip_bytes": 1075597174,
            "zip_sha256": "21001ac266c465f5d345ec42d7b508c6a6328"
                          "487fd9d4d7774422dd5ea10ad83",
            "zip_sha256_reverified_wave9": True,
            "nesting": f"zip -> IMS.7z -> {test_dir_name}.rar -> "
                       f"{n} timestamped snapshot files",
            "extracted_scope": f"{test_dir_name} extracted in wave 9 "
                               f"from the local archive retained "
                               f"since wave 5 (no re-download needed)",
        },
        "ground_truth": meta[
            {"2nd_test": "ground_truth_test2",
             "3rd_test": "ground_truth_test3"}[test_dir_name]],
        "reference_test1_wave5": {
            "record": "ims__taskC__isolation_forest__s42",
            "bearing3_first_sustained_alarm": "1819 / 2156 "
                                              "(337 before run end)",
            "bearing4_first_sustained_alarm": "1433 / 2156 "
                                              "(723 before run end)",
            "note": "Test 1 detectors used 10 features per bearing "
                    "(2 channels); Tests 2-3 use 5 (1 channel) — "
                    "alarm timings are comparable as protocol "
                    "outcomes, not as identical-feature results",
        },
    }
    save_record(rec, {
        "n_snapshots": int(n),
        f"bearing{failed_bearing}_first_sustained_alarm":
            per_bearing[failed_key]["first_sustained_alarm_index"],
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", default="all",
                        choices=["test2", "test3", "all"])
    args = parser.parse_args()
    if args.set in ("test2", "all"):
        run_ims_test("test2")
    if args.set in ("test3", "all"):
        run_ims_test("test3")
