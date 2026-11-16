# PREDICAI backend on Hugging Face Spaces (Docker SDK, free CPU tier).
# The Rust edge API (Axum + ONNX Runtime): live RUL predictions at
# POST /predict_rul, health at GET /health, dashboard data at
# GET /data/demo and GET /data/platform.
# Layout of this Space repo: edge/ (Rust source), artifacts/ (ONNX model
# + preprocessing), data/ (exported dashboard JSONs).

# --- ONNX Runtime shared library (same pip wheel as the local setup:
# --- libonnxruntime 1.32.0, proven with ort 2.0.0-rc.15 load-dynamic) ---
FROM python:3.12-slim AS ortlib
RUN pip download onnxruntime==1.32.0 --no-deps -d /tmp/wheel \
 && python -c "import zipfile, glob; zipfile.ZipFile(glob.glob('/tmp/wheel/*.whl')[0]).extractall('/tmp/wheel/x')" \
 && mkdir -p /ort \
 && cp /tmp/wheel/x/onnxruntime/capi/libonnxruntime.so.1.32.0 /ort/libonnxruntime.so

# --- Build the Rust API ---
FROM rust:1-bookworm AS build
WORKDIR /src
COPY edge/Cargo.toml edge/Cargo.lock ./edge/
COPY edge/src ./edge/src
WORKDIR /src/edge
RUN cargo build --release

# --- Runtime ---
FROM debian:bookworm-slim
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 libstdc++6 ca-certificates \
 && rm -rf /var/lib/apt/lists/* \
 && useradd -m app \
 && mkdir -p /app/artifacts /app/data /app/lib
WORKDIR /app
COPY --from=build /src/edge/target/release/predicai-edge /app/predicai-edge
COPY --from=ortlib /ort/libonnxruntime.so /app/lib/libonnxruntime.so
COPY artifacts/cmapss_lstm.onnx artifacts/preprocessing.json /app/artifacts/
COPY data/demo_data.json data/platform_data.json /app/data/
RUN chown -R app:app /app
ENV ORT_DYLIB_PATH=/app/lib/libonnxruntime.so \
    PREDICAI_MODEL=/app/artifacts/cmapss_lstm.onnx \
    PREDICAI_PREPROCESSING=/app/artifacts/preprocessing.json \
    PREDICAI_DB=/tmp/predicai.db \
    PREDICAI_DEMO_DATA=/app/data/demo_data.json \
    PREDICAI_PLATFORM_DATA=/app/data/platform_data.json \
    PREDICAI_BIND=0.0.0.0 \
    PORT=8080
USER app
EXPOSE 8080
CMD ["/app/predicai-edge"]
