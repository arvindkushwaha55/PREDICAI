# PREDICAI backend: the Rust edge API (Axum + ONNX Runtime) as a public
# web service, so the hosted dashboard has a real backend: live model
# predictions at POST /predict_rul, health at GET /health, and the
# dashboard's data served from GET /data/demo and GET /data/platform.
#
# Build context is the REPO ROOT (this file needs edge/, ml/artifacts/
# and dashboard/src/data/). Render reads render.yaml at the repo root,
# which points here.

# --- ONNX Runtime shared library -------------------------------------
# Taken from the same pip wheel the local setup runs on
# (libonnxruntime 1.32.0, proven with ort 2.0.0-rc.15 load-dynamic).
FROM python:3.12-slim AS ortlib
RUN pip download onnxruntime==1.32.0 --no-deps -d /tmp/wheel \
 && python -c "import zipfile, glob; zipfile.ZipFile(glob.glob('/tmp/wheel/*.whl')[0]).extractall('/tmp/wheel/x')" \
 && mkdir -p /ort \
 && cp /tmp/wheel/x/onnxruntime/capi/libonnxruntime.so.1.32.0 /ort/libonnxruntime.so

# --- Build the Rust API ----------------------------------------------
FROM rust:1-bookworm AS build
WORKDIR /src
COPY edge/Cargo.toml edge/Cargo.lock ./edge/
COPY edge/src ./edge/src
COPY edge/tests ./edge/tests
WORKDIR /src/edge
RUN cargo build --release

# --- Runtime ----------------------------------------------------------
FROM debian:bookworm-slim
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 libstdc++6 ca-certificates \
 && rm -rf /var/lib/apt/lists/* \
 && useradd -m app \
 && mkdir -p /app/data /app/artifacts /app/lib
WORKDIR /app
COPY --from=build /src/edge/target/release/predicai-edge /app/predicai-edge
COPY --from=ortlib /ort/libonnxruntime.so /app/lib/libonnxruntime.so
COPY ml/artifacts/cmapss_lstm.onnx ml/artifacts/preprocessing.json /app/artifacts/
COPY dashboard/src/data/demo_data.json dashboard/src/data/platform_data.json /app/data/
RUN chown -R app:app /app
ENV ORT_DYLIB_PATH=/app/lib/libonnxruntime.so \
    PREDICAI_MODEL=/app/artifacts/cmapss_lstm.onnx \
    PREDICAI_PREPROCESSING=/app/artifacts/preprocessing.json \
    PREDICAI_DB=/app/data/predicai.db \
    PREDICAI_DEMO_DATA=/app/data/demo_data.json \
    PREDICAI_PLATFORM_DATA=/app/data/platform_data.json \
    PREDICAI_BIND=0.0.0.0
USER app
EXPOSE 8080
CMD ["/app/predicai-edge"]
