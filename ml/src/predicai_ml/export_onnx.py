"""PREDICAI — ONNX export + parity verification (Part 7 / Phase A6).

Used after the Kaggle training run returned best_model.pt:
- rebuilds the exact architecture from model.py, loads the weights
- re-checks PyTorch predictions against the Kaggle test_predictions.npy
  (proves the .pt is the model that produced metrics.json)
- exports cmapss_lstm.onnx (opset 17, dynamic batch, LEGACY exporter —
  dynamo=False; torch>=2.9's dynamo default needs onnxscript, which was
  the one failure in the Kaggle run)
- verifies PyTorch vs ONNX Runtime outputs on fixed windows (<= 1e-4)

Run:  python -m predicai_ml.export_onnx --pt <best_model.pt> --out <dir>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from predicai_ml.data import load_fd001
from predicai_ml.model import LSTMRULRegressor
from predicai_ml.preprocessing import FEATURES, run_pipeline


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", required=True)
    ap.add_argument("--kaggle-preds", default=None,
                    help="test_predictions.npy from the Kaggle run (engine, true, pred)")
    ap.add_argument("--raw-dir", default="ml/data/raw")
    ap.add_argument("--out", default="ml/artifacts")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    model = LSTMRULRegressor(len(FEATURES), 128, 64, 32, 0.2)
    state = torch.load(args.pt, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    # Rebuild the exact test windows locally (pipeline is deterministic —
    # preprocessing.json from Kaggle was verified identical to the local one).
    train, test, rul = load_fd001(args.raw_dir)
    res = run_pipeline(train, test, rul)
    x = torch.from_numpy(res.X_test_last)

    with torch.no_grad():
        pt_pred = torch.clamp(model(x), min=0.0).numpy()

    if args.kaggle_preds:
        kag = np.load(args.kaggle_preds)  # engine, y_true, y_pred
        order = np.argsort(kag[:, 0])
        kag_pred = kag[order, 2]
        diff = float(np.max(np.abs(pt_pred - kag_pred)))
        print(f"PyTorch(here) vs Kaggle predictions: max abs diff = {diff:.6f}")
        assert diff < 0.05, "predictions diverge — wrong weights or pipeline"
        print("OK: this .pt reproduces the Kaggle metrics' predictions")

    # ---- ONNX export (legacy exporter) ----
    onnx_path = out / "cmapss_lstm.onnx"
    dummy = torch.randn(1, 30, len(FEATURES))
    torch.onnx.export(
        model, dummy, str(onnx_path),
        input_names=["input"], output_names=["rul"],
        dynamic_axes={"input": {0: "batch"}, "rul": {0: "batch"}},
        opset_version=17, do_constant_folding=True, dynamo=False,
    )
    import onnx
    onnx.checker.check_model(onnx.load(str(onnx_path)))
    print(f"ONNX exported + checker OK: {onnx_path}")

    # ---- Parity: PyTorch vs ONNX Runtime on the same 100 test windows ----
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        raw_pt = model(x).numpy()
    raw_ox = sess.run(None, {"input": res.X_test_last})[0].reshape(-1)
    max_diff = float(np.max(np.abs(raw_pt - raw_ox)))
    print(f"PyTorch vs ONNX parity: max abs diff = {max_diff:.2e} (tolerance 1e-4)")
    assert max_diff <= 1e-4, "PARITY FAILED"
    print("PARITY OK")


if __name__ == "__main__":
    main()
