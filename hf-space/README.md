---
title: PREDICAI API
emoji: "🔧"
colorFrom: gray
colorTo: blue
sdk: docker
app_port: 8080
pinned: false
license: mit
---

# PREDICAI backend API

The Rust edge API for the PREDICAI predictive-maintenance dashboard
(https://arvindkushwaha55.github.io/PREDICAI/).

- `GET /health` — server status and whether the ONNX model is loaded
- `POST /predict_rul` — live Remaining Useful Life prediction (cycles)
  from a 30 x 14 sensor window, served by the shipped FD001 LSTM model
- `GET /data/demo` and `GET /data/platform` — the dashboard's exported
  research data

Source: https://github.com/arvindkushwaha55/PREDICAI (MIT). Research and
education demo; predictions are not validated for real machinery.
