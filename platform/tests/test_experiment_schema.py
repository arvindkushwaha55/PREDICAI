"""Every experiment record in experiments/ must carry the full schema
from ARCHITECTURE.md §4 — dataset, task, model, config, seed, split
protocol, metrics, train/inference timing, library versions, git commit.
A record missing any of these is not reportable.
"""
import json

import pytest

from conftest import PLATFORM_DIR

REQUIRED_KEYS = {
    "experiment_id", "timestamp_utc", "dataset", "task", "model",
    "model_params", "seed", "split", "preprocessing", "metrics",
    "timing", "environment", "git_commit",
}

EXP_DIR = PLATFORM_DIR / "experiments"
RECORDS = sorted(p for p in EXP_DIR.glob("*.json") if p.name != "INDEX.json")


@pytest.mark.skipif(not RECORDS, reason="no experiments have been run yet")
@pytest.mark.parametrize("path", RECORDS, ids=[p.stem for p in RECORDS])
def test_experiment_record_schema(path):
    rec = json.loads(path.read_text())
    assert REQUIRED_KEYS <= set(rec), f"missing keys: {REQUIRED_KEYS - set(rec)}"
    assert rec["task"] in ("A", "B", "C")
    assert rec["seed"] == 42
    assert rec["split"], "split protocol must be recorded"
    assert rec["metrics"], "metrics must be recorded"
    assert "train_s" in rec["timing"]
    assert "infer_ms_per_row" in rec["timing"]
    libs = rec["environment"]["libraries"]
    for pkg in ("numpy", "pandas", "scikit-learn"):
        assert libs[pkg], f"library version missing: {pkg}"
    assert rec["git_commit"] not in ("", "unknown")


@pytest.mark.skipif(not RECORDS, reason="no experiments have been run yet")
def test_index_matches_records():
    index = json.loads((EXP_DIR / "INDEX.json").read_text())
    indexed = {e["experiment_id"] for e in index["experiments"]}
    on_disk = {p.stem for p in RECORDS}
    assert indexed == on_disk
