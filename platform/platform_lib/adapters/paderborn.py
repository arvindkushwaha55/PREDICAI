"""Paderborn University Bearing adapter (KAt-DataCenter) — Task A.

Source: Chair of Design and Drive Technology (KAt), Paderborn University,
Bearing DataCenter. Downloaded 2026-11-15 from the official directory
listing linked by the official download page — no gate, no form:
    https://groups.uni-paderborn.de/kat/BearingDataCenter/<CODE>.rar
    landing: https://mb.uni-paderborn.de/en/kat/research/bearing-datacenter
Licence (verbatim on the official page): Creative Commons
Attribution-NonCommercial 4.0 International (CC BY-NC 4.0);
"Non-commercial academic use of the data is permitted, but a citation of
the source is required" (Lessmeier et al., KAt-DataCenter). NON-COMMERCIAL
ONLY — do not use this dataset in any commercial context.
Archive SHA-256 (as downloaded):
    K001.rar  0f119ebdb28fb2f4d9fac1beb1319429f63f7ae1256c23c872f280f3560918e5
    KA01.rar  6a6be1e11132730cc6f560d51eacedcbfd5fd74b829e9d8d3728c6c8a7cd4c0e
    KI01.rar  b1dd6d99bb64d556f889fefaedb7e6e672900f5f015125615feaae776f055348
    KA04.rar  6fc317157e2cd4cb55bc54d6fac1b1753f8ab31abcccd1ac68cbfb78e5e8299d
    KI04.rar  8fde861a9ceac4b41fe8a030b4c5bbb4b9b9af0590d9338a8de0150b4661be77
    KI14.rar  16911b5f8c723b568b6aa69397b00796fea41e43bf627a1a0158a7294f2b7a60
              (downloaded wave 10; profile read, archive not extracted —
              see the wave-10 runner for why KI04 was preferred)

Integrated scope: 3 of the 32 bearings by DEFAULT —
    K001  healthy (no damage)
    KA01  outer-race damage, artificial (EDM-seeded)
    KI01  inner-race damage, artificial (EDM-seeded)
Wave 10 additionally downloaded the REAL-damage bearings KA04 / KI04
(damage origin verified verbatim from the official per-bearing damage
profile PDFs shipped inside each archive — KA04: mode "fatigue",
symptom "Pitting", component OR, damage method "lifetime test";
KI04: primary damage fatigue pitting on IR by lifetime test, plus a
secondary documented OR plastic-deformation damage). They are in
BEARINGS below but NOT in the default load: pass them explicitly via
the `bearings` argument so the wave-3/4 scope is unchanged.
Each bearing archive holds 80 measurement files: 4 operating conditions
x 20 measurements of 4 s. Condition codes decode per the dataset
documentation: N15/N09 = 1500/900 rpm, M07/M01 = 0.7/0.1 Nm torque,
F10/F04 = 1000/400 N radial force.

Each .mat holds one struct (named after the file) whose Y array carries
7 channels; this adapter uses `vibration_1` (64 kHz, 256,001 samples per
4 s measurement) and windows it exactly like the CWRU adapter
(2,048-sample windows, same 10 time-domain features) so the two bearing
datasets share a feature space. The synchronously recorded phase
currents (also 64 kHz) are present in the files and noted in metadata
for future electrical-fault work; they are not featurised here.

The .rar archives must be extracted first (e.g. with 7-Zip: `7zz x
K001.rar`); the adapter reads <raw_dir>/<BEARING>/*.mat.

Split caveat (same family as CWRU, one level better): there are 20
measurements per condition per bearing, but only ONE physical bearing
per class in this scope — evaluation must split by measurement file at
minimum, and class-level generalisation to unseen bearings cannot be
claimed from 3 bearings. The metadata states this.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

from .base import DatasetAdapter
from .cwru import FEATURE_COLUMNS, window_features

WINDOW_LEN = 2048
SAMPLE_RATE_HZ = 64_000
CHANNEL = "vibration_1"

# bearing code -> (class label, damage origin)
BEARINGS = {
    "K001": ("Normal", "none (healthy bearing)"),
    "KA01": ("OuterRace", "artificial (EDM-seeded)"),
    "KI01": ("InnerRace", "artificial (EDM-seeded)"),
    # Wave 10 — real fatigue damage (official damage-profile PDFs in
    # each archive; see adapter docstring). Not in the default load.
    "KA04": ("OuterRace", "real (fatigue pitting, lifetime test)"),
    "KI04": ("InnerRace", "real (fatigue pitting, lifetime test; "
                           "secondary OR indentations documented)"),
}

# The wave-3/4 integrated scope — the default load, unchanged.
DEFAULT_BEARINGS = ("K001", "KA01", "KI01")
LABELS = ["Normal", "InnerRace", "OuterRace"]

_CONDITION_RE = re.compile(r"^(N\d+)_M(\d+)_F(\d+)$")


def parse_condition(code: str) -> dict:
    """Decode a Paderborn operating-condition code, e.g. N15_M07_F10."""
    m = _CONDITION_RE.match(code)
    if not m:
        return {"condition": code}
    speed = int(m.group(1)[1:]) * 100        # N15 -> 1500 rpm
    torque = int(m.group(2)) / 10.0          # M07 -> 0.7 Nm
    force = int(m.group(3)) * 100            # F10 -> 1000 N
    return {"condition": code, "speed_rpm_nominal": speed,
            "torque_nm_nominal": torque, "radial_force_n_nominal": force}


def _unwrap(x):
    while isinstance(x, np.ndarray) and x.dtype == object and x.size == 1:
        x = x.ravel()[0]
    return x


def _text(x) -> str:
    x = _unwrap(x)
    if isinstance(x, np.ndarray):
        x = x.ravel()[0]
    return str(x)


def load_vibration(path) -> np.ndarray:
    """Extract the vibration_1 channel (64 kHz) from one measurement."""
    mat = loadmat(path, struct_as_record=False)
    keys = [k for k in mat if not k.startswith("__")]
    if len(keys) != 1:
        raise ValueError(f"{path}: expected one top-level struct, "
                         f"found {keys}")
    struct = _unwrap(mat[keys[0]])
    for ch in np.asarray(struct.Y).ravel():
        ch = _unwrap(ch)
        if _text(ch.Name) == CHANNEL:
            return np.asarray(ch.Data, dtype=np.float64).ravel()
    raise ValueError(f"{path}: channel {CHANNEL!r} not found")


class PaderbornAdapter(DatasetAdapter):
    name = "paderborn"

    def __init__(self, raw_dir: str | Path,
                 bearings: tuple[str, ...] = DEFAULT_BEARINGS) -> None:
        super().__init__(raw_dir)
        unknown = set(bearings) - set(BEARINGS)
        if unknown:
            raise ValueError(f"unknown Paderborn bearings: {unknown}")
        self.bearings = bearings

    def load(self) -> tuple[pd.DataFrame, dict]:
        rows, per_bearing = [], []
        for code in self.bearings:
            label, origin = BEARINGS[code]
            files = sorted((self.raw_dir / code).glob("*.mat"))
            if not files:
                raise FileNotFoundError(
                    f"no .mat files under {self.raw_dir / code} — "
                    f"download {code}.rar from the official KAt "
                    f"DataCenter listing and extract it (see adapter "
                    f"docstring)")
            n_windows = 0
            for path in files:
                series = load_vibration(path)
                cond = parse_condition(path.name.split(f"_{code}_")[0])
                for i in range(len(series) // WINDOW_LEN):
                    feats = window_features(
                        series[i * WINDOW_LEN:(i + 1) * WINDOW_LEN])
                    feats.update(cond)
                    feats["label"] = label
                    feats["bearing"] = code
                    feats["damage_origin"] = origin
                    feats["source_file"] = path.name
                    rows.append(feats)
                    n_windows += 1
            per_bearing.append({
                "bearing": code, "label": label, "damage_origin": origin,
                "n_measurements": int(len(files)),
                "n_windows": int(n_windows),
            })
        df = pd.DataFrame(rows)
        metadata = {
            "dataset": "paderborn",
            "source": "Paderborn University KAt-DataCenter, Bearing "
                      "DataCenter (Lessmeier et al. 2016); CC BY-NC 4.0",
            "licence": "CC BY-NC 4.0 (non-commercial; citation required)",
            "tasks": ["A"],
            "target": "label (bearing condition diagnosed from a "
                      "2048-sample vibration window)",
            "maintenance_history": "absent",
            "data_nature": "real test-rig recordings; damage origin "
                           "per bearing as listed in `bearings` "
                           "(artificial EDM-seeded for KA01/KI01; real "
                           "fatigue pitting from lifetime tests for "
                           "KA04/KI04, per the official damage-profile "
                           "PDFs shipped in each archive)",
            "feature_columns": FEATURE_COLUMNS,
            "channel": CHANNEL,
            "window_len": WINDOW_LEN,
            "sample_rate_hz": SAMPLE_RATE_HZ,
            "also_in_files": "phase_current_1/2 (64 kHz), speed, torque, "
                             "force, temperature — not featurised here",
            "bearings": per_bearing,
            "n_windows": int(len(df)),
            "split_caveat": "one physical bearing per class in this "
                            "scope; split by measurement file at "
                            "minimum — no unseen-bearing generalisation "
                            "claim is possible from 3 bearings",
        }
        return df, metadata
