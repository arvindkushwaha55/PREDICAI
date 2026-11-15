"""PREDICAI: Part 11 / Phase F: dashboard data export.

Builds `src/data/demo_data.json` for the React dashboard from REAL project
artifacts only. Nothing is invented:

- Replay series: `simulator/demo.db` -> `predictions_log` (rows written by
  the Rust edge API itself during Part 10 simulator replays of FD001 test
  engines 1 / 49 / 100, source = 'RUST EDGE'). The Rust API exposes no
  history endpoint, so the dashboard consumes this static export instead.
- Cycle alignment: the simulator sends one prediction per recorded cycle
  from cycle 30 onward, so prediction #k (in log order) is the window
  ENDING at recorded cycle 29 + k. This is VERIFIED below against the raw
  FD001 test file (logged count must equal total_cycles - 29 per engine);
  if the check ever fails, cycles are exported as null rather than guessed.
- True RUL per cycle: RUL_FD001[engine] + (total_cycles - cycle), the same
  definition as preprocessing/simulator.
- Maintenance: demo.db ships with EMPTY maintenance tables, so the records
  are produced at export time by the Part 8 maintenance module itself
  (`seed_synthetic_demo()` on a throwaway DB; demo.db is opened read-only
  and never modified), plus one recommendation computed by the module's own
  `make_recommendation()` from ENGINE-001's REAL final replay prediction.
  Every record keeps its is_synthetic flag.
- Model info: `ml/artifacts/model_metadata.json`, `metrics.json`,
  `explainability.json` (real Kaggle training run, 13 Nov 2026).
- Live sample window: `edge/tests/fixtures/parity_windows.json`, engine 1
  case: the exact raw 30x14 window the Rust parity test uses.

Run:  ../.venv/bin/python data_export.py
"""
from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
SEQ_LEN = 30
ENGINES = (1, 49, 100)

sys.path.insert(0, str(REPO / "maintenance"))


def machine_id(engine: int) -> str:
    return f"ENGINE-{engine:03d}"


def load_test_cycles_and_rul() -> tuple[dict[int, int], dict[int, float]]:
    """Total recorded cycles per test engine + published final RUL."""
    raw = REPO / "ml" / "data" / "raw"
    totals: dict[int, int] = {}
    for line in (raw / "test_FD001.txt").read_text().splitlines():
        if line.strip():
            eng = int(float(line.split()[0]))
            totals[eng] = totals.get(eng, 0) + 1
    ruls = {}
    for i, line in enumerate(
        (raw / "RUL_FD001.txt").read_text().splitlines(), start=1
    ):
        if line.strip():
            ruls[i] = float(line.split()[0])
    return totals, ruls


def export_engines(db: sqlite3.Connection) -> list[dict]:
    totals, final_ruls = load_test_cycles_and_rul()
    out = []
    for eng in ENGINES:
        mid = machine_id(eng)
        rows = db.execute(
            "SELECT pred_id, created_at, predicted_rul, source "
            "FROM predictions_log WHERE machine_id = ? ORDER BY pred_id",
            (mid,),
        ).fetchall()
        total = totals[eng]
        final_true = final_ruls[eng]
        # Verify the cycle alignment instead of assuming it (see docstring).
        aligned = len(rows) == total - (SEQ_LEN - 1)
        series = []
        for k, (_pid, created_at, pred, _src) in enumerate(rows, start=1):
            end_cycle = (SEQ_LEN - 1) + k if aligned else None
            series.append(
                {
                    "step": k,
                    "end_cycle": end_cycle,
                    "predicted_rul": pred,
                    "true_rul": (
                        final_true + (total - end_cycle)
                        if end_cycle is not None
                        else None
                    ),
                    "created_at": created_at,
                }
            )
        preds = [r[2] for r in rows]
        errs = [
            abs(p["predicted_rul"] - p["true_rul"])
            for p in series
            if p["true_rul"] is not None
        ]
        out.append(
            {
                "engine": eng,
                "machine_id": mid,
                "source": (
                    "FD001 REPLAY: predictions logged by the Rust edge API "
                    "(predictions_log.source = 'RUST EDGE') during the Part 10 "
                    "simulator replay of NASA C-MAPSS FD001 test engine "
                    f"{eng}; exported from simulator/demo.db"
                ),
                "total_recorded_cycles": total,
                "predictions_logged": len(rows),
                "cycle_alignment_verified": aligned,
                "final": {
                    "predicted_rul": preds[-1],
                    "true_rul": final_true,
                    "abs_error": abs(preds[-1] - final_true),
                },
                "stats": {
                    "min_predicted_rul": min(preds),
                    "max_predicted_rul": max(preds),
                    "mean_predicted_rul": sum(preds) / len(preds),
                    "mean_abs_error_vs_true": (
                        sum(errs) / len(errs) if errs else None
                    ),
                },
                "series": series,
            }
        )
    return out


def export_maintenance(final_pred_rul: float) -> dict:
    """Part 8 module generates its own seeded demo + one recommendation."""
    from maintenance import MaintenanceDB  # maintenance/maintenance.py

    with tempfile.TemporaryDirectory() as tmp:
        db = MaintenanceDB(Path(tmp) / "export.db")
        db.seed_synthetic_demo()
        # The synthetic demo machine's reference "current cycle" is 440,
        # the same value the Part 8 module's own tests use
        # (maintenance/test_maintenance.py). The predicted RUL is real:
        # ENGINE-001's final replay prediction from simulator/demo.db.
        db.make_recommendation(
            "ENGINE-001", predicted_rul=final_pred_rul, current_cycle=440
        )
        machine = db.get_machine("ENGINE-001")
        records = db.service_history("ENGINE-001")
        recs = [
            dict(r)
            for r in db.conn.execute(
                "SELECT * FROM recommendations WHERE machine_id = 'ENGINE-001' "
                "ORDER BY rec_id"
            ).fetchall()
        ]
        db.close()
    return {
        "machine_id": "ENGINE-001",
        "source": (
            "SYNTHETIC DEMO DATA: FD001 contains no real service records. "
            "Generated at export time by the Part 8 maintenance module "
            "(seed_synthetic_demo + policy v1.0) on a throwaway database; "
            "simulator/demo.db itself was not modified."
        ),
        "machine": machine,
        "service_records": records,
        "recommendations": recs,
        "recommendation_inputs": {
            "predicted_rul": final_pred_rul,
            "predicted_rul_source": (
                "ENGINE-001 final prediction logged in simulator/demo.db "
                "(real model output)"
            ),
            "current_cycle": 440,
            "current_cycle_source": (
                "synthetic demo machine reference cycle, as used by the "
                "Part 8 module's own tests"
            ),
        },
    }


def export_model() -> dict:
    art = REPO / "ml" / "artifacts"
    meta = json.loads((art / "model_metadata.json").read_text())
    metrics = json.loads((art / "metrics.json").read_text())
    xai = json.loads((art / "explainability.json").read_text())
    return {
        "source": (
            "ml/artifacts/model_metadata.json + metrics.json + "
            "explainability.json: real Kaggle training run (owner, "
            "13 Nov 2026); no placeholder numbers"
        ),
        "architecture": meta["architecture"],
        "model_file": meta["model_file"],
        "weights": meta["weights"],
        "dataset": meta["dataset"],
        "seq_len": meta["seq_len"],
        "num_features": meta["num_features"],
        "features": xai["data"]["features"],
        "rul_cap": meta["rul_cap"],
        "seed": meta["seed"],
        "onnx_opset": meta["onnx_opset"],
        "trained_on": meta["trained_on"],
        "training_config": metrics["config"],
        "metrics": {
            "test_last_window_per_engine": metrics[
                "test_last_window_per_engine"
            ],
            "validation_all_windows": metrics["val_all_windows"],
            "best_val_rmse_during_training": metrics[
                "best_val_rmse_during_training"
            ],
        },
        "parity": meta["parity"],
        "explainability": {
            "note": xai["honesty_note"],
            "split": xai["data"]["split"],
            "n_val_windows": xai["data"]["n_val_windows"],
            "permutation_top5": [
                {"sensor": r["sensor"], "rmse_increase": r["rmse_increase"]}
                for r in xai["permutation_importance"]["ranking"][:5]
            ],
            "integrated_gradients_top5": [
                {
                    "sensor": r["sensor"],
                    "mean_abs_attribution": r["mean_abs_attribution"],
                    "mean_signed_attribution": r["mean_signed_attribution"],
                }
                for r in xai["integrated_gradients"]["ranking"][:5]
            ],
            "top5_overlap": xai["sanity_checks"]["top5_overlap_between_methods"],
        },
    }


def export_live_sample() -> dict:
    fx = json.loads(
        (REPO / "edge" / "tests" / "fixtures" / "parity_windows.json").read_text()
    )
    case = next(c for c in fx["cases"] if c["engine"] == 1)
    return {
        "source": (
            "edge/tests/fixtures/parity_windows.json (engine 1 case): raw "
            "30x14 sensor window from real FD001 test data; the same window "
            "the Rust parity test uses"
        ),
        "machine_id": "ENGINE-001",
        "window": case["raw_window"],
        "expected_rul_from_parity_fixture": case["expected_rul"],
        "true_rul": case["true_rul_clipped"],
    }



def _no_em_dash(obj):
    """Display normalisation: the dashboard style bans em dashes
    (U+2014) in displayed text. Any string copied from an upstream
    record or module (protocol, notes, verdicts, maintenance text)
    is rewritten to the spaced-hyphen form here, at export time.
    The source records and modules are never edited."""
    if isinstance(obj, str):
        return obj.replace(" \u2014 ", " - ").replace("\u2014", "-")
    if isinstance(obj, list):
        return [_no_em_dash(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _no_em_dash(v) for k, v in obj.items()}
    return obj

def main() -> None:
    db = sqlite3.connect(f"file:{REPO / 'simulator' / 'demo.db'}?mode=ro", uri=True)
    engines = export_engines(db)
    db.close()

    data = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "unit": "cycles",
        "policy_thresholds": {
            "source": (
                "maintenance/policy.py (policy version 1.0): POLICY CHOICES "
                "for planning reviews, NOT scientifically validated failure "
                "boundaries"
            ),
            "review_max_rul": 30,
            "inspection_max_rul": 50,
            "service_review_cycles": 300,
        },
        "engines": engines,
        "maintenance": export_maintenance(engines[0]["final"]["predicted_rul"]),
        "model": export_model(),
        "live_sample": export_live_sample(),
    }
    data = _no_em_dash(data)
    dest = HERE / "src" / "data" / "demo_data.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(data, indent=1))
    print(f"wrote {dest} ({dest.stat().st_size/1024:.0f} KiB)")
    for e in engines:
        print(
            f"  {e['machine_id']}: {e['predictions_logged']} predictions, "
            f"final pred {e['final']['predicted_rul']:.3f} vs true "
            f"{e['final']['true_rul']:.0f}, aligned={e['cycle_alignment_verified']}"
        )


if __name__ == "__main__":
    main()
