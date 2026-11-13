"""PREDICAI — preprocessing tests (Part 3 gate). Toy engines + real FD001."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from predicai_ml.data import load_fd001  # noqa: E402
from predicai_ml.preprocessing import (  # noqa: E402
    FEATURES, add_test_rul, add_train_rul, build_windows, clip_rul,
    engine_split, fit_scaler, run_pipeline,
)

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"


def toy_engine(unit: int, life: int, base: float = 0.0) -> pd.DataFrame:
    rows = {"unit": [unit] * life, "cycle": list(range(1, life + 1))}
    for f in FEATURES:
        rows[f] = [base + c * 0.1 for c in range(1, life + 1)]
    return pd.DataFrame(rows)


def test_train_rul_formula():
    df = add_train_rul(toy_engine(1, 5))
    assert df["rul"].tolist() == [4, 3, 2, 1, 0]


def test_test_rul_formula():
    # engine observed for 5 cycles, true RUL at last cycle = 10
    df = add_test_rul(toy_engine(1, 5), np.array([10]))
    assert df["rul"].tolist() == [14, 13, 12, 11, 10]


def test_rul_clip():
    df = clip_rul(add_train_rul(toy_engine(1, 200)), cap=125)
    assert df["rul"].max() == 125
    assert df["rul"].min() == 0


def test_engine_split_disjoint_and_complete():
    train_e, val_e = engine_split(np.arange(1, 101))
    assert set(train_e).isdisjoint(val_e)
    assert set(train_e) | set(val_e) == set(range(1, 101))
    assert len(val_e) == 20 and len(train_e) == 80


def test_windows_shape_and_no_engine_mixing():
    df = pd.concat([toy_engine(1, 40, 0.0), toy_engine(2, 35, 100.0)])
    df = add_train_rul(df)
    X, y, engines = build_windows(df, seq_len=30)
    assert X.shape == (11 + 6, 30, len(FEATURES))
    # engine 2 windows must contain its own (offset) values only
    e2 = X[engines == 2]
    assert np.all(e2 > 50)
    # label = RUL at last cycle of the window
    assert y[0] == 40 - 30


def test_scaler_fit_on_train_only():
    train, _, _ = load_fd001(RAW)
    train = add_train_rul(train)
    tr_e, _ = engine_split(train["unit"].unique())
    part = train[train["unit"].isin(tr_e)]
    scaler = fit_scaler(part)
    expected_mean = part[FEATURES].to_numpy().mean(axis=0)
    assert np.allclose(scaler.mean, expected_mean)
    scaled = scaler.transform(part)
    assert np.allclose(scaled[FEATURES].to_numpy().mean(axis=0), 0, atol=1e-9)


def test_full_pipeline_shapes_real_data():
    train, test, rul = load_fd001(RAW)
    res = run_pipeline(train, test, rul)
    f = len(FEATURES)
    assert res.X_train.shape[1:] == (30, f)
    assert res.X_val.shape[1:] == (30, f)
    assert res.X_test_last.shape == (100, 30, f)   # one window per test engine
    assert res.y_test_last.shape == (100,)
    assert res.y_train.max() <= 125 and res.y_train.min() >= 0
