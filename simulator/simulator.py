"""PREDICAI — Part 10 / Phase E: FD001 replay simulator.

SIMULATED REPLAY — not live sensors. This replays recorded NASA C-MAPSS
FD001 *test* data, one engine cycle by cycle, exactly as it was recorded.
From cycle 30 onward it sends the last 30 cycles of RAW sensor values
(the 14 model features, unnormalized) to the Rust edge API
(POST /predict_rul), which normalizes, runs the ONNX model and logs every
prediction into SQLite `predictions_log` itself (source = 'RUST EDGE').

True RUL at any replayed cycle (same definition as preprocessing):
    true_rul(cycle) = RUL_FD001[engine] + (total_observed_cycles - cycle)
so at the engine's last observed cycle, true RUL == RUL_FD001[engine].

Usage:
    python simulator.py --engine 1                 # full replay, max speed
    python simulator.py --engine 49 --speed 5      # 5 predictions/second
    python simulator.py --engine 100 --api http://localhost:18081

The Rust API must already be running (see simulator/README.md).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

# Repo layout: this file lives in <repo>/simulator/, the ML package in
# <repo>/ml/src. Make imports work no matter where it is invoked from.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "ml" / "src"))

from predicai_ml.data import load_fd001  # noqa: E402
from predicai_ml.preprocessing import FEATURES  # noqa: E402

SEQ_LEN = 30
SOURCE_LABEL = "SIMULATED REPLAY (NASA FD001 test data)"


# --------------------------------------------------------------------------
# Pure helpers (unit-tested in test_simulator.py)
# --------------------------------------------------------------------------
def true_rul_at(final_rul: float, total_cycles: int, current_cycle: int) -> float:
    """True remaining cycles at `current_cycle` (1-based).

    `final_rul` is the published RUL_FD001 value: the true RUL at the
    engine's LAST observed cycle. Every earlier cycle has proportionally
    more life left.
    """
    return float(final_rul) + (total_cycles - current_cycle)


def iter_windows(
    rows: np.ndarray, seq_len: int = SEQ_LEN
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (end_cycle, window) for every full window in `rows`.

    `rows` is (n_cycles, n_features) in recorded order. `end_cycle` is the
    1-based cycle number of the window's last row. The first window ends at
    cycle `seq_len` — nothing is yielded before a full window exists (no
    padding is invented). Each window holds the LAST `seq_len` raw rows in
    recorded order.
    """
    for end in range(seq_len, len(rows) + 1):
        yield end, rows[end - seq_len : end]


def machine_id_for(engine: int) -> str:
    """Machine id used consistently by the API's predictions_log."""
    return f"ENGINE-{engine:03d}"


# --------------------------------------------------------------------------
# API client (stdlib only)
# --------------------------------------------------------------------------
def api_health(api: str) -> dict:
    with urllib.request.urlopen(f"{api}/health", timeout=10) as resp:
        return json.loads(resp.read().decode())


def api_predict(api: str, machine_id: str, window: np.ndarray) -> tuple[float, float]:
    """POST one raw window; return (predicted_rul, latency_seconds)."""
    body = json.dumps(
        {"machine_id": machine_id, "window": window.tolist()}
    ).encode()
    req = urllib.request.Request(
        f"{api}/predict_rul", data=body, headers={"Content-Type": "application/json"}
    )
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=30) as resp:
        payload = json.loads(resp.read().decode())
    latency = time.perf_counter() - start
    if payload.get("status") != "ok" or payload.get("unit") != "cycles":
        raise RuntimeError(f"unexpected API response: {payload}")
    return float(payload["predicted_rul"]), latency


# --------------------------------------------------------------------------
# Replay
# --------------------------------------------------------------------------
def replay(
    engine: int,
    api: str,
    speed: float,
    max_cycles: int | None,
    raw_dir: str = "ml/data/raw",
) -> dict:
    """Replay one FD001 test engine against the API. Returns a summary dict."""
    _train, test, rul_test = load_fd001(str(REPO_ROOT / raw_dir))
    final_rul = float(rul_test[engine - 1])  # RUL_FD001 is 1-indexed by engine

    group = test[test["unit"] == engine].sort_values("cycle")
    if group.empty:
        raise ValueError(f"engine {engine} not present in FD001 test data")
    rows = group[FEATURES].to_numpy(dtype=np.float64)
    cycles = group["cycle"].to_numpy()
    total_cycles = len(group)
    if max_cycles is not None:
        rows, cycles = rows[:max_cycles], cycles[:max_cycles]

    machine_id = machine_id_for(engine)
    print(f"[{SOURCE_LABEL}] engine {engine} -> {machine_id}: "
          f"{len(rows)} recorded cycles, true RUL at last cycle = {final_rul:.0f}")

    errors: list[float] = []
    latencies: list[float] = []
    last_pred = last_true = None
    for end_cycle, window in iter_windows(rows):
        # end_cycle is a position in the replayed slice; the recorded cycle
        # number is cycles[end_cycle - 1] (identical for FD001 test data,
        # which is contiguous from cycle 1, but read it from the data).
        recorded_cycle = int(cycles[end_cycle - 1])
        true_rul = true_rul_at(final_rul, total_cycles, recorded_cycle)
        pred, latency = api_predict(api, machine_id, window)
        errors.append(abs(pred - true_rul))
        latencies.append(latency)
        last_pred, last_true = pred, true_rul
        n = len(errors)
        if n % 25 == 0:
            print(f"  cycle {recorded_cycle}: predictions={n} "
                  f"pred={pred:.2f} true={true_rul:.0f} |err|={abs(pred - true_rul):.2f}")
        if speed > 0:
            time.sleep(max(0.0, 1.0 / speed - latency))

    if not errors:
        raise RuntimeError(
            f"engine {engine}: only {len(rows)} cycles replayed — "
            f"no full {SEQ_LEN}-cycle window, so no predictions were made"
        )

    summary = {
        "engine": engine,
        "machine_id": machine_id,
        "cycles_replayed": len(rows),
        "predictions": len(errors),
        "final_predicted_rul": last_pred,
        "final_true_rul": last_true,
        "replay_mae": float(np.mean(errors)),
        "mean_latency_ms": float(np.mean(latencies) * 1000.0),
    }
    print(f"[{SOURCE_LABEL}] DONE engine {engine}: "
          f"cycles={summary['cycles_replayed']} predictions={summary['predictions']} "
          f"final pred={last_pred:.3f} vs true={last_true:.0f} cycles | "
          f"replay MAE={summary['replay_mae']:.2f} cycles | "
          f"mean API latency={summary['mean_latency_ms']:.1f} ms (client-side)")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--engine", type=int, default=1, help="FD001 test engine (default 1)")
    ap.add_argument("--speed", type=float, default=0.0,
                    help="predictions per second pacing; 0 = as fast as possible (default)")
    ap.add_argument("--api", default="http://localhost:18081", help="Rust API base URL")
    ap.add_argument("--max-cycles", type=int, default=None,
                    help="replay at most this many cycles of the engine")
    args = ap.parse_args(argv)

    try:
        health = api_health(args.api)
    except (urllib.error.URLError, OSError) as e:
        print(f"ERROR: Rust API not reachable at {args.api} ({e}). "
              f"Start it first — see simulator/README.md.", file=sys.stderr)
        return 1
    if not health.get("model_loaded"):
        print(f"ERROR: API is up but the ONNX model is not loaded: {health}",
              file=sys.stderr)
        return 1

    try:
        replay(args.engine, args.api, args.speed, args.max_cycles)
    except (urllib.error.URLError, OSError, RuntimeError, ValueError) as e:
        print(f"ERROR: replay failed: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
