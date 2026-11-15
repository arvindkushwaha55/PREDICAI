"""Case Western Reserve University (CWRU) Bearing Data Center adapter.

Task A (fault *diagnosis*): classify a vibration window as Normal /
InnerRace / Ball / OuterRace. This wave integrates exactly four
recordings — one per class, all 12 kHz drive-end (DE), 0 HP load,
0.007-inch EDM-seeded faults where applicable:

    file      CWRU id   condition                    DE samples
    97.mat    normal_0  healthy baseline             243,938
    105.mat   IR007_0   inner-race fault 0.007"      121,265
    118.mat   B007_0    ball fault 0.007"            122,571
    130.mat   OR007@6_0 outer-race fault 0.007"      121,991
                        (centred @ 6:00 position)

Provenance: downloaded 2026-11-15 from the official Bearing Data Center
file host (https://engineering.case.edu/sites/default/files/<id>.mat),
linked from the official "12k Drive End Bearing Fault Data" and "Normal
Baseline Data" pages. Each file's identity is self-verified by the
MATLAB variable inside it (X097_DE_time, X105_DE_time, ...), and file
sizes match the Data Center's published catalogue.

The adapter turns each recording's DE series into non-overlapping
windows of WINDOW_LEN samples and one row of time-domain statistics per
window (the classic CWRU feature set). KNOWN LIMITATION, binding for
every score produced from this adapter: there is ONE recording per
class, so any window-wise split puts windows of the same recording on
both sides — scores measure within-recording separability, not
generalisation to unseen bearings/recordings. The experiment record
states this; do not quote CWRU numbers without it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.stats import kurtosis, skew

from .base import DatasetAdapter

WINDOW_LEN = 2048
SAMPLE_RATE_HZ = 12_000

# local filename -> (CWRU condition id, class label)
RECORDINGS = {
    "97.mat": ("normal_0", "Normal"),
    "105.mat": ("IR007_0", "InnerRace"),
    "118.mat": ("B007_0", "Ball"),
    "130.mat": ("OR007@6_0", "OuterRace"),
}
LABELS = ["Normal", "InnerRace", "Ball", "OuterRace"]

FEATURE_COLUMNS = [
    "mean", "std", "rms", "skewness", "kurtosis", "crest_factor",
    "peak", "min", "max", "peak_to_peak",
]


def window_features(window: np.ndarray) -> dict:
    """Time-domain statistics of one vibration window."""
    x = np.asarray(window, dtype=np.float64)
    rms = float(np.sqrt(np.mean(x ** 2)))
    peak = float(np.max(np.abs(x)))
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "rms": rms,
        "skewness": float(skew(x)),
        "kurtosis": float(kurtosis(x)),  # Fisher (excess) kurtosis
        "crest_factor": float(peak / rms) if rms > 0 else 0.0,
        "peak": peak,
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "peak_to_peak": float(np.max(x) - np.min(x)),
    }


def load_de_series(path) -> np.ndarray:
    """Extract the drive-end time series from a CWRU .mat file.

    The DE variable is named X<file-id>_DE_time; the id embedded in the
    variable name is also returned by load() for provenance checking.
    """
    mat = loadmat(path)
    de_keys = [k for k in mat if k.endswith("_DE_time")]
    if len(de_keys) != 1:
        raise ValueError(f"{path}: expected exactly one *_DE_time "
                         f"variable, found {de_keys}")
    return np.asarray(mat[de_keys[0]], dtype=np.float64).ravel()


class CWRUAdapter(DatasetAdapter):
    name = "cwru"

    def load(self) -> tuple[pd.DataFrame, dict]:
        rows, recordings_meta = [], []
        for fname, (condition, label) in RECORDINGS.items():
            path = self.raw_dir / fname
            if not path.exists():
                raise FileNotFoundError(
                    f"CWRU recording missing: {path} — download the "
                    f"official file for {condition} from the CWRU "
                    f"Bearing Data Center (see adapter docstring)")
            series = load_de_series(path)
            n_windows = len(series) // WINDOW_LEN
            for i in range(n_windows):
                feats = window_features(
                    series[i * WINDOW_LEN:(i + 1) * WINDOW_LEN])
                feats["label"] = label
                feats["source_file"] = fname
                rows.append(feats)
            recordings_meta.append({
                "file": fname, "condition": condition, "label": label,
                "n_samples": int(len(series)), "n_windows": int(n_windows),
                "official_url": "https://engineering.case.edu/sites/"
                                f"default/files/{fname}",
            })
        df = pd.DataFrame(rows)
        metadata = {
            "dataset": "cwru",
            "source": "Case Western Reserve University Bearing Data "
                      "Center, 12 kHz drive-end data + normal baseline "
                      "(https://engineering.case.edu/bearingdatacenter/"
                      "download-data-file)",
            "tasks": ["A"],
            "target": "label (bearing condition diagnosed from a "
                      "2048-sample vibration window)",
            "maintenance_history": "absent",
            "data_nature": "real test-rig recordings; faults are "
                           "EDM-seeded single-point defects on a 2 hp "
                           "motor rig, not field failures",
            "feature_columns": FEATURE_COLUMNS,
            "window_len": WINDOW_LEN,
            "sample_rate_hz": SAMPLE_RATE_HZ,
            "recordings": recordings_meta,
            "n_windows": int(len(df)),
            "split_caveat": "one recording per class; window-wise splits "
                            "share recordings across train/test and are "
                            "optimistic — see adapter docstring",
        }
        return df, metadata
