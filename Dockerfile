# Hybrid Token-Efficient Routing Agent
# CPU-only, self-contained; runs anywhere Docker runs (Render, laptops, scoring env).
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# curl: model download at start. build tools: llama-cpp-python compiles from source.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl build-essential cmake ninja-build \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Portable llama.cpp build: GGML_NATIVE=OFF so the binary runs on any x86-64
# host (build machine != runtime machine on PaaS builders), and capped
# parallelism so the compile is not OOM-killed on small build machines.
ENV FORCE_CMAKE=1 \
    CMAKE_ARGS="-DGGML_NATIVE=OFF" \
    CMAKE_BUILD_PARALLEL_LEVEL=4

# Dependency layer: install with an empty src so llama.cpp only recompiles
# when pyproject changes, not on every code push (20 min -> ~1 min deploys).
COPY pyproject.toml README.md ./
RUN mkdir -p src && pip install ".[all]"

# App layer: real sources, reinstalled without touching dependencies.
COPY src ./src
RUN pip install --no-deps --force-reinstall .

COPY config.yaml ./
COPY tasks ./tasks
COPY scripts/entrypoint.sh /entrypoint.sh
# World-writable model/data dirs: HF Spaces (and other PaaS) run as non-root.
# /input and /output: the judging harness mounts over these; creating them
# writable lets local `docker run -v` tests work identically.
RUN chmod +x /entrypoint.sh && mkdir -p models data /input /output \
    && chmod -R 777 models data /input /output

# Optional: bake the local GGUF into the image (recommended for submission -
# the container then starts with zero network fetches, well inside the 60s
# readiness rule). Build with:
#   docker build --build-arg MODEL_URL=https://.../gemma-3-1b-it-q4_0.gguf .
ARG MODEL_URL=""
RUN if [ -n "$MODEL_URL" ]; then \
        curl -fSL --retry 3 -o models/gemma-3-1b-it-q4_0.gguf "$MODEL_URL"; \
    fi
ENV HOME=/tmp

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s \
    CMD curl -fsS "http://localhost:${PORT:-8000}/health" || exit 1

ENTRYPOINT ["/entrypoint.sh"]
