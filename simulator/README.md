# PREDICAI Simulator — SIMULATED REPLAY (Part 10 / Phase E)

> **What this is:** a replay of *recorded* NASA C-MAPSS FD001 **test** data.
> It is **not** live sensor data and no real machine is connected. Every run
> is labelled `SIMULATED REPLAY (NASA FD001 test data)`, and the predictions
> the Rust API logs for it carry `source = 'RUST EDGE'` in SQLite — they are
> never presented as real sensor readings.

## What it does

`simulator.py` takes one FD001 test engine and replays its recorded cycles
in order. From cycle 30 onward, each step sends the **last 30 cycles of raw
sensor values** (the model's 14 features, unnormalized — the API normalizes
using `preprocessing.json`) to the Rust edge API:

```
POST /predict_rul   {"machine_id": "ENGINE-001", "window": [[...14 values...] x 30]}
```

The API runs the ONNX LSTM, returns the predicted Remaining Useful Life in
**cycles**, and logs the prediction into `predictions_log` itself. The
simulator tracks the true RUL at every cycle
(`RUL_FD001[engine] + total_cycles − current_cycle`) and reports the replay
MAE and the client-side mean API latency at the end.

## Run it

1. Start the Rust API (from the repo's `edge/` directory). The
   `ORT_DYLIB_PATH` variable is **required** — the API is built with
   onnxruntime's `load-dynamic` feature and will start with
   `model_loaded: false` without it:

   ```bash
   cd edge
   ORT_DYLIB_PATH=$HOME/workspace/predicai/.venv/lib/python3.12/site-packages/onnxruntime/capi/libonnxruntime.so.1.32.0 \
   PREDICAI_PORT=18081 \
   PREDICAI_DB=$HOME/workspace/predicai/simulator/demo.db \
   cargo run        # or ./target/debug/predicai-edge
   ```

   Check `GET http://localhost:18081/health` shows `"model_loaded":true`.

2. Replay an engine (from the repo root):

   ```bash
   .venv/bin/python simulator/simulator.py --engine 1            # full speed
   .venv/bin/python simulator/simulator.py --engine 49 --speed 5 # paced, 5 pred/s
   .venv/bin/python simulator/simulator.py --engine 100 --max-cycles 60
   ```

   Options: `--engine N` (default 1), `--speed S` predictions/second
   (default 0 = as fast as possible), `--api URL`
   (default `http://localhost:18081`), `--max-cycles N`.

3. Inspect what was logged (the demo database used above):

   ```bash
   sqlite3 simulator/demo.db \
     "SELECT machine_id, COUNT(*), source FROM predictions_log GROUP BY 1, 3;"
   ```

## Reading the output

- **cycles replayed** — recorded cycles the engine has in FD001 test data
  (varies per engine: engine 1 has 31, engine 49 has 303, engine 100 has 198).
- **predictions** — one per cycle from cycle 30 on (`cycles − 29`); earlier
  cycles have no full 30-cycle window, so no prediction is made and no
  padding is invented.
- **final predicted vs true RUL** — at the last recorded cycle the true RUL
  is exactly the published `RUL_FD001` value, so this is directly comparable
  to the official test evaluation.
- **replay MAE** — mean |predicted − true| across *all* predicted cycles of
  the replay. This is **not** the headline test metric: the model was trained
  with RUL capped at 125 cycles, so early in a long engine's life (true RUL
  far above 125) predictions sit near the cap and the error is large by
  construction. The official metric uses only each engine's last window.
- **mean API latency** — client-side round-trip per prediction, measured by
  the simulator (includes local HTTP); it is a local observation, not a
  benchmark claim.

## Tests

```bash
cd ~/workspace/predicai
.venv/bin/python -m pytest simulator/test_simulator.py -q
```

Unit tests cover the pure helpers only (true-RUL formula, window slicing,
machine-id format). The end-to-end path is verified by an actual replay run
against the API plus a row count in `predictions_log`.
