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

# Optional: bake the local GGUF into the image (recommended for submission -
# the container then starts with zero network fetches, well inside the 60s
# readiness rule). Placed BEFORE the source layer so code changes never
# re-download the 700MB model. Build with:
#   docker build --build-arg MODEL_URL=https://.../gemma-3-1b-it-q4_0.gguf .
ARG MODEL_URL=""
# Prefer a GGUF baked from the build context (see the .dockerignore exception);
# fall back to MODEL_URL only when no local file was copied. Baking the local
# copy sidesteps the gated-HuggingFace 401 (google/gemma-* needs a token).
# The model[s] glob keeps COPY from erroring when the file is not in context.
COPY model[s] ./models
RUN mkdir -p models && if [ ! -f models/gemma-3-1b-it-q4_0.gguf ] && [ -n "$MODEL_URL" ]; then \
        curl -fSL --retry 3 -o models/gemma-3-1b-it-q4_0.gguf "$MODEL_URL"; \
    fi

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
# APP_ROOT anchors config.yaml and models/ so they resolve regardless of the
# working directory the judging harness starts us in (a CWD-relative default
# loses both at once, and the run then ships empty answers). Declared late so
# it cannot invalidate the cached llama.cpp build or the baked model layer.
ENV HOME=/tmp \
    APP_ROOT=/app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s \
    CMD curl -fsS "http://localhost:${PORT:-8000}/health" || exit 1

ENTRYPOINT ["/entrypoint.sh"]
