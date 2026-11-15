"""PREDICAI — Phase 8 (Wave 7): platform data export for the dashboard.

Builds `src/data/platform_data.json` from the platform experiment store.
NOTHING is invented here either:

- Experiments / transfers / XAI: read at export time from
  `platform/experiments/INDEX.json` + every record JSON it lists. Every
  number in the output traces to a record file on disk; the script fails
  loudly if a listed record is missing or unparseable.
- Datasets: the DATASETS table below is copied faithfully from
  `platform/DATASET_REGISTRY.md` (the operational registry, verified
  2026-11-15, Waves 1-3 + wave-5 IMS update). If the registry changes,
  update the registry first, then this table — the registry is the
  source, this table is a transcription for the dashboard.
- "Best result" rows are computed from the records with the metric named
  on the row. One deliberate editorial rule, carried from the platform
  reports: the wave-2 CWRU window-split record (accuracy 1.000) is
  EXCLUDED from best-result computation — its own notes call it
  non-generalisable; the recording-wise record is the CWRU headline.

Run:  ../.venv/bin/python platform_export.py
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
EXP_DIR = REPO / "platform" / "experiments"

# ---------------------------------------------------------------------------
# Dataset facts — SOURCE: platform/DATASET_REGISTRY.md (verified 2026-11-15).
# Fields per entry: id (matches the `dataset` key used in experiment
# records where one exists), name, source, licence, tasks (A/B/C),
# status, adapter, note. Licence strings quote/summarise the registry;
# "not confirmed / not stated" is preserved verbatim in spirit — it does
# NOT mean public domain.
# ---------------------------------------------------------------------------
DATASETS: list[dict] = [
    {
        "id": "cmapss_fd001",
        "name": "C-MAPSS FD001 — Turbofan Degradation",
        "source": "NASA Ames PCoE (PHM mirror)",
        "licence": "NASA research use; formal licence text not confirmed",
        "tasks": ["B"],
        "status": "integrated",
        "adapter": "CMapssAdapter + ml/ pipeline",
        "note": "1 operating condition, 1 fault mode (HPC). Train 20,631 rows / 100 engines. The original PREDICAI system (LSTM → ONNX → Rust API) serves this subset.",
    },
    {
        "id": "cmapss_fd002",
        "name": "C-MAPSS FD002 — Turbofan Degradation",
        "source": "NASA Ames PCoE (PHM mirror)",
        "licence": "NASA research use; formal licence text not confirmed",
        "tasks": ["B"],
        "status": "integrated",
        "adapter": "CMapssAdapter",
        "note": "6 operating conditions, 1 fault mode. Train 53,759 rows / 260 engines; 23 features after derivation.",
    },
    {
        "id": "cmapss_fd003",
        "name": "C-MAPSS FD003 — Turbofan Degradation",
        "source": "NASA Ames PCoE (PHM mirror)",
        "licence": "NASA research use; formal licence text not confirmed",
        "tasks": ["B"],
        "status": "integrated",
        "adapter": "CMapssAdapter",
        "note": "1 operating condition, 2 fault modes (HPC + fan). Train 24,720 rows / 100 engines; 15 features (sensor_6 kept, unlike FD001).",
    },
    {
        "id": "cmapss_fd004",
        "name": "C-MAPSS FD004 — Turbofan Degradation",
        "source": "NASA Ames PCoE (PHM mirror)",
        "licence": "NASA research use; formal licence text not confirmed",
        "tasks": ["B"],
        "status": "integrated",
        "adapter": "CMapssAdapter",
        "note": "6 operating conditions, 2 fault modes. Train 61,249 rows / 249 engines. On FD004 the LSTM wins RMSE/MAE but loses the NASA score to Ridge — recorded as measured.",
    },
    {
        "id": "ai4i2020",
        "name": "AI4I 2020 Predictive Maintenance",
        "source": "UCI ML Repository",
        "licence": "CC BY 4.0",
        "tasks": ["A", "C"],
        "status": "integrated",
        "adapter": "AI4IAdapter",
        "note": "10,000 × 14; binary machine-failure label (339 failures, ≈3.4%). Data is SYNTHETIC (rule-generated failures) — the 5 failure-mode flags define the target and are declared leakage columns, never features.",
    },
    {
        "id": "secom",
        "name": "SECOM Manufacturing",
        "source": "UCI ML Repository",
        "licence": "CC BY 4.0",
        "tasks": ["A", "C"],
        "status": "integrated",
        "adapter": "SECOMAdapter",
        "note": "1,567 rows × 590 anonymised sensor columns; 104 fails (≈6.6%). Heavy missingness → 450 features survive the wave-1 pipeline. Features are anonymised — explanations cite sensor indices only.",
    },
    {
        "id": "cwru",
        "name": "CWRU Bearing",
        "source": "Case Western Reserve University",
        "licence": "Not stated on official page",
        "tasks": ["A"],
        "status": "integrated",
        "adapter": "CWRUAdapter",
        "note": "Drive-end vibration at 12 kHz; Normal / InnerRace / Ball / OuterRace. Fault DIAGNOSIS of seeded faults, not prediction. Headline result is the recording-wise evaluation (accuracy 0.665) — see Research view.",
    },
    {
        "id": "paderborn",
        "name": "Paderborn Bearing",
        "source": "KAt-DataCenter, Paderborn University",
        "licence": "CC BY-NC 4.0 (confirmed on official page)",
        "tasks": ["A"],
        "status": "integrated",
        "adapter": "PaderbornAdapter",
        "note": "Integrated scope: 3 of 32 bearings (K001 healthy, KA01 outer-race, KI01 inner-race) → 30,022 windows. Non-commercial licence. One physical bearing per class — no unseen-bearing generalisation claim.",
    },
    {
        "id": "ims",
        "name": "IMS Bearing (Run-to-Failure)",
        "source": "NASA PCoE / University of Cincinnati",
        "licence": "Not stated on PCoE page",
        "tasks": ["C", "B"],
        "status": "integrated",
        "adapter": "IMSAdapter",
        "note": "Test 1 integrated (wave 5, after the user approved the 1.08 GB download): 2,156 snapshots × 8 channels at 20 kHz. Teardown ground truth: bearing 3 inner-race + bearing 4 roller defects. No per-snapshot labels — evaluation is alarm timing only.",
    },
    {
        "id": "nasa_battery",
        "name": "NASA Li-ion Battery Aging",
        "source": "NASA Ames PCoE",
        "licence": "No named licence; acknowledgement requested",
        "tasks": ["B"],
        "status": "integrated",
        "adapter": "NASABatteryAdapter",
        "note": "4 room-temperature run-to-failure batteries (B0005/6/7/18), 636 discharge-cycle rows. EOL = capacity < 1.4 Ah; B0007 never crosses (censored) and is excluded from LOBO scoring.",
    },
    {
        "id": "femto_pronostia",
        "name": "FEMTO-ST / PRONOSTIA Bearing",
        "source": "FEMTO-ST / NASA PCoE mirrors",
        "licence": "Not confirmed",
        "tasks": ["B"],
        "status": "researched",
        "adapter": None,
        "note": "17 run-to-failure bearings (IEEE PHM 2012). Researched-only: ingestion effort (thousands of small CSVs) and mirror provenance pinning put it behind current priorities.",
    },
    {
        "id": "xjtu_sy",
        "name": "XJTU-SY Bearing",
        "source": "Xi'an Jiaotong University (researcher page)",
        "licence": "Not stated",
        "tasks": ["B"],
        "status": "researched",
        "adapter": None,
        "note": "15 run-to-failure bearings, ≈4.4 GB. Researched-only: mirror-only distribution (provenance risk) and unstated licence.",
    },
    {
        "id": "mfpt",
        "name": "MFPT Bearing Fault Data",
        "source": "MFPT Society (mfpt.org)",
        "licence": "Not stated (live page gone)",
        "tasks": ["A"],
        "status": "blocked",
        "adapter": None,
        "note": "BLOCKED — the official site is gone: mfpt.org redirects domain-wide to ASNT, which offers no dataset, download or licence. No unofficial mirror was substituted.",
    },
    {
        "id": "mimii",
        "name": "MIMII Industrial Machine Sound",
        "source": "Hitachi / Zenodo",
        "licence": "CC BY-SA 4.0",
        "tasks": ["C"],
        "status": "researched",
        "adapter": None,
        "note": "Researched-only BY DECISION: 100.2 GB of audio (12 zips) is disproportionate to the platform's current stage. A future wave could take one machine type at one SNR (≈7–11 GB).",
    },
    {
        "id": "bosch",
        "name": "Bosch Production Line Performance",
        "source": "Kaggle (Bosch)",
        "licence": "Kaggle competition rules",
        "tasks": ["A"],
        "status": "blocked",
        "adapter": None,
        "note": "BLOCKED — no direct download exists; access requires a Kaggle account and acceptance of competition rules (≈14.3 GB, 4,264 anonymised features, ≈0.58% failures).",
    },
    {
        "id": "milling",
        "name": "Milling Tool Wear",
        "source": "UC Berkeley BEST Lab / NASA",
        "licence": "NASA portal; formal licence text not confirmed",
        "tasks": ["A", "B"],
        "status": "researched",
        "adapter": None,
        "note": "167 milling runs; label = measured flank wear VB (mm). Some runs have missing VB and a few are flagged corrupted in the dataset's own notes — to be excluded and documented, never silently repaired.",
    },
    {
        "id": "hydraulic",
        "name": "Hydraulic Condition Monitoring",
        "source": "University of Duisburg-Essen / UCI",
        "licence": "CC BY 4.0",
        "tasks": ["A"],
        "status": "researched",
        "adapter": None,
        "note": "2,205 cycles × 17 sensors; multi-output graded condition classes (cooler / valve / pump leakage / accumulator) — condition grades, not failure events. Not run-to-failure.",
    },
    {
        "id": "metropt3",
        "name": "MetroPT-3 (Lisbon Metro APU)",
        "source": "Metro do Porto / UCI",
        "licence": "CC BY 4.0",
        "tasks": ["C"],
        "status": "researched",
        "adapter": None,
        "note": "1,516,948 rows × 15 features, Jan–Aug 2020. Ground truth = timestamped maintenance reports only, no per-row labels — evaluation must be event-based.",
    },
    {
        "id": "automotive_engine",
        "name": "Automotive Engine Health",
        "source": "—",
        "licence": "—",
        "tasks": [],
        "status": "not-found",
        "adapter": None,
        "note": "No verified public sensor/time-series automotive engine-health dataset with a clear licence was found (searched 2026-11-15). The category is recorded as unfilled rather than inventing an entry.",
    },
    {
        "id": "motor_electrical",
        "name": "Motor Electrical Faults (candidates)",
        "source": "UCI / Mendeley / 4TU.ResearchData",
        "licence": "CC BY 4.0 / CC0 (verified per candidate)",
        "tasks": ["A"],
        "status": "researched",
        "adapter": None,
        "note": "No single canonical dataset. Licence-verified candidates researched: UCI Sensorless Drive Diagnosis (CC BY 4.0), KAIST PMSM stator faults (CC BY 4.0), UOEMD-VAFCVS (CC BY 4.0), NLN-EMP (CC0). None downloaded yet.",
    },
]

# The wave-2 CWRU record whose own notes forbid quoting it as
# generalisation performance (window split over 4 recordings).
CWRU_WINDOW_SPLIT_RECORD = "cwru__taskA__random_forest__s42"


def record_kind(entry: dict, record: dict) -> str:
    model = entry.get("model", "")
    if model.startswith("transfer_"):
        return "transfer"
    if model.startswith("shap_") or model.startswith("permutation_"):
        return "explainability"
    return "experiment"


def primary_metrics(record: dict) -> dict:
    """Flatten the record's primary metric block to a small dict."""
    metrics = record.get("metrics") or {}
    out: dict = {}
    if "test_last_window_per_engine" in metrics:
        blk = metrics["test_last_window_per_engine"]
        for k in ("rmse", "mae", "nasa_score"):
            if k in blk:
                out[k] = blk[k]
    if "pooled" in metrics:
        out["pooled_rmse"] = metrics["pooled"]["rmse"]
        out["pooled_mae"] = metrics["pooled"]["mae"]
    for k in ("accuracy", "f1", "f1_macro", "precision", "recall",
              "roc_auc", "false_alarm_rate"):
        if k in metrics:
            out[k] = metrics[k]
    return out


def reference_blocks(extra: dict) -> list[dict]:
    """All in-domain reference blocks a transfer record carries.

    Both the source-side and target-side in-domain results are kept —
    picking just one would hide half of what the record documents.
    """
    out = []
    for key, val in extra.items():
        if key.startswith("reference") and isinstance(val, dict):
            out.append({
                "label": key.removeprefix("reference_").replace("_", " "),
                **{k: val[k] for k in ("rmse", "mae", "nasa_score",
                                       "accuracy", "f1_macro")
                   if k in val},
            })
    return out


def degradation_block(extra: dict) -> dict | None:
    for key, val in extra.items():
        if key.startswith("degradation") and isinstance(val, dict):
            return {"label": key, **val}
    if "accuracy_delta_vs_target_in_domain" in extra:
        return {"accuracy_delta_vs_target_in_domain":
                extra["accuracy_delta_vs_target_in_domain"]}
    return None


# Verdicts for the Research view — one line each, written from the
# records' own notes + degradation blocks (numbers re-checked at export
# time against the record; the line is a summary, the record is truth).
TRANSFER_VERDICTS = {
    "cmapss_fd001_to_fd003__taskB__transfer_lstm__s42":
        "Not deployable — cross-domain RMSE 48.80 vs 15.04 in-domain on "
        "FD003 (+224%). FD003's second (fan) fault mode was never seen "
        "in FD001 training.",
    "cmapss_fd003_to_fd001__taskB__transfer_cnn_lstm__s42":
        "Near in-domain — RMSE 15.82 vs 13.73 shipped in-domain (+15%). "
        "Wide→narrow transfer (2 fault modes → 1) largely works; the "
        "reverse direction does not.",
    "cmapss_fd002_to_fd004__taskB__transfer_ridge__s42":
        "Fails — RMSE 51.38 vs 19.43 in-domain Ridge on FD004 (+165%), "
        "despite an identical 23-feature space. The shift is in the "
        "distributions / fault modes, not the features.",
    "cmapss_fd004_to_fd002__taskB__transfer_ridge__s42":
        "Near in-domain — RMSE 18.07 vs 16.58 in-domain Ridge on FD002 "
        "(+9%). Again the wide→narrow direction (2 fault modes → 1).",
    "cwru_to_paderborn__taskA__transfer_rf__s42":
        "Fails — accuracy 0.253 vs 0.950 in-domain. The Normal class is "
        "never recognised on the target rig (recall 0.0): 'Normal' is "
        "rig-specific with time-domain features.",
    "paderborn_to_cwru__taskA__transfer_rf__s42":
        "Fails — accuracy 0.125 vs 0.665 recording-wise in-domain. "
        "Normal recall 0.0 again; 12 kHz vs 64 kHz sampling also makes "
        "the same 2048-sample window cover 170.7 ms vs 32.0 ms.",
}


def xai_ranking(record: dict) -> tuple[list[dict], str]:
    """Extract a feature ranking from an explainability record's extra."""
    extra = record.get("extra") or {}
    for key, val in extra.items():
        if isinstance(val, list) and val and isinstance(val[0], dict) \
                and "feature" in val[0]:
            ranking = []
            for row in val:
                value = None
                label = ""
                for vk in ("mean_abs_shap", "importance_mean"):
                    if vk in row:
                        value = row[vk]
                        label = vk
                        break
                ranking.append({
                    "feature": row["feature"],
                    "value": value,
                    "value_label": label,
                })
            return ranking, key
    return [], ""


def main() -> None:
    index = json.loads((EXP_DIR / "INDEX.json").read_text())
    entries = index["experiments"]

    experiments: list[dict] = []
    transfers: list[dict] = []
    xai: list[dict] = []
    records_by_id: dict[str, dict] = {}

    for entry in entries:
        rec_path = EXP_DIR / entry["file"]
        record = json.loads(rec_path.read_text())
        records_by_id[entry["experiment_id"]] = record
        kind = record_kind(entry, record)
        split = record.get("split") or {}
        timing = record.get("timing") or {}
        base = {
            "id": entry["experiment_id"],
            "dataset": entry["dataset"],
            "task": entry["task"],
            "model": entry["model"],
            "kind": kind,
            "seed": record.get("seed"),
            "timestamp_utc": entry.get("timestamp_utc"),
            "headline": entry.get("headline") or {},
            "train_s": timing.get("train_s"),
            "infer_ms_per_row": timing.get("infer_ms_per_row"),
            "protocol": split.get("protocol"),
            "notes": record.get("notes"),
        }
        experiments.append(base)

        if kind == "transfer":
            extra = record.get("extra") or {}
            transfers.append({
                **base,
                "source_dataset": split.get("source_dataset"),
                "target_dataset": split.get("target_dataset"),
                "cross_metrics": primary_metrics(record),
                "in_domain_references": reference_blocks(extra),
                "degradation": degradation_block(extra),
                "verdict": TRANSFER_VERDICTS.get(entry["experiment_id"]),
            })
        elif kind == "explainability":
            ranking, ranking_key = xai_ranking(record)
            xai.append({
                **base,
                "method": ("SHAP" if entry["model"].startswith("shap_")
                           else "Permutation importance"),
                "ranking": ranking,
                "ranking_key": ranking_key,
            })

    # ---- counts -----------------------------------------------------------
    task_counts: dict[str, int] = {}
    for e in experiments:
        task_counts[e["task"]] = task_counts.get(e["task"], 0) + 1
    counts = {
        "records": len(experiments),
        "transfers": len(transfers),
        "explainability": len(xai),
        "trained_model_records": sum(
            1 for e in experiments if e["kind"] == "experiment"),
        "datasets_integrated": sum(
            1 for d in DATASETS if d["status"] == "integrated"),
        "datasets_registry_entries": len(DATASETS),
        "tasks": task_counts,
    }
    assert counts["records"] == len(entries), "INDEX/record count mismatch"

    # ---- per-dataset record counts ---------------------------------------
    per_dataset_counts: dict[str, int] = {}
    for e in experiments:
        per_dataset_counts[e["dataset"]] = \
            per_dataset_counts.get(e["dataset"], 0) + 1
    datasets_out = [
        {**d, "records": per_dataset_counts.get(d["id"], 0)}
        for d in DATASETS
    ]

    # ---- best result per dataset x task (computed, metric named) ---------
    best: list[dict] = []

    def add_best(dataset: str, task: str, metric_label: str,
                 candidates: list[dict], key: str, lower: bool,
                 display: str | None = None) -> None:
        pool = [c for c in candidates
                if c["dataset"] == dataset and c["task"] == task
                and c["kind"] == "experiment"
                and c["id"] != CWRU_WINDOW_SPLIT_RECORD
                and isinstance(c["headline"].get(key), (int, float))]
        if not pool:
            return
        winner = (min if lower else max)(
            pool, key=lambda c: c["headline"][key])
        best.append({
            "dataset": dataset,
            "task": task,
            "experiment_id": winner["id"],
            "model": winner["model"],
            "metric_label": metric_label,
            "metric_value": winner["headline"][key],
            "metric_display": display,
            "protocol": winner["protocol"],
        })

    exp_only = experiments
    for ds in ("ai4i2020", "secom"):
        add_best(ds, "A", "F1 (test)", exp_only, "f1", lower=False)
    for ds in ("cwru", "paderborn"):
        add_best(ds, "A", "F1-macro", exp_only, "f1_macro", lower=False)
    for ds in ("cmapss_fd001", "cmapss_fd002", "cmapss_fd003",
               "cmapss_fd004"):
        add_best(ds, "B", "Test RMSE (cycles)", exp_only, "test_rmse",
                 lower=True)
    add_best("nasa_battery", "B", "Pooled LOBO RMSE (cycles)", exp_only,
             "pooled_rmse", lower=True)
    add_best("secom", "C", "F1 (test)", exp_only, "f1", lower=False)
    # IMS has no per-snapshot labels, so no F1 exists to rank by; its
    # best-row is the alarm-timing summary taken from the record itself.
    ims_rec = records_by_id.get("ims__taskC__isolation_forest__s42")
    if ims_rec is not None:
        at = ims_rec["metrics"]["alarm_timing"]
        b3, b4 = at["bearing_3"], at["bearing_4"]
        best.append({
            "dataset": "ims",
            "task": "C",
            "experiment_id": "ims__taskC__isolation_forest__s42",
            "model": "isolation_forest",
            "metric_label": "Alarm timing (no per-snapshot labels exist)",
            "metric_value": None,
            "metric_display": (
                f"Bearing 3 (inner-race defect): first sustained alarm "
                f"{b3['snapshots_of_warning_before_run_end']} snapshots "
                f"before run end · Bearing 4 (roller defect): "
                f"{b4['snapshots_of_warning_before_run_end']} snapshots "
                f"before end · Bearings 1–2 (no defect): no sustained "
                f"alarm"),
            "protocol": ims_rec["split"]["protocol"],
        })

    # ---- IMS alarm summary for the Research view --------------------------
    ims_out = None
    if ims_rec is not None:
        at = ims_rec["metrics"]["alarm_timing"]
        ims_out = {
            "experiment_id": "ims__taskC__isolation_forest__s42",
            "n_snapshots": ims_rec["split"]["n_snapshots"],
            "framing": ims_rec["metrics"]["framing"],
            "protocol": ims_rec["split"]["protocol"],
            "bearings": [
                {
                    "bearing": name.replace("_", " "),
                    "documented_defect": vals["documented_defect"],
                    "first_sustained_alarm_index":
                        vals["first_sustained_alarm_index"],
                    "first_sustained_alarm_timestamp":
                        vals["first_sustained_alarm_timestamp"],
                    "snapshots_of_warning_before_run_end":
                        vals["snapshots_of_warning_before_run_end"],
                    "fraction_of_run_before_alarm":
                        vals["fraction_of_run_before_alarm"],
                    "last_decile_exceedance_rate":
                        vals["last_decile_exceedance_rate"],
                }
                for name, vals in at.items()
            ],
        }

    data = {
        "generated_at": datetime.now(timezone.utc).isoformat(
            timespec="seconds"),
        "source": (
            "platform/experiments/INDEX.json + the record JSONs it lists "
            "(read at export time) and platform/DATASET_REGISTRY.md "
            "(dataset facts transcribed in platform_export.py). No "
            "placeholder numbers."
        ),
        "counts": counts,
        "datasets": datasets_out,
        "experiments": experiments,
        "transfers": transfers,
        "xai": xai,
        "best_results": best,
        "ims_alarms": ims_out,
    }
    dest = HERE / "src" / "data" / "platform_data.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(data, indent=1))
    print(f"wrote {dest} ({dest.stat().st_size/1024:.0f} KiB)")
    print(f"  records={counts['records']} transfers={counts['transfers']} "
          f"xai={counts['explainability']} "
          f"trained={counts['trained_model_records']}")
    print(f"  datasets integrated={counts['datasets_integrated']}/"
          f"{counts['datasets_registry_entries']}  tasks={task_counts}")
    print(f"  best rows={len(best)}  ims={'yes' if ims_out else 'MISSING'}")


if __name__ == "__main__":
    main()
