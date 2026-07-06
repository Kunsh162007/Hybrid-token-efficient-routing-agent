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

COPY pyproject.toml README.md ./
COPY src ./src

# Full install: local inference + semantic cache + learned router + web.
RUN pip install ".[all]"

COPY config.yaml ./
COPY tasks ./tasks
COPY scripts/entrypoint.sh /entrypoint.sh
# World-writable model/data dirs: HF Spaces (and other PaaS) run as non-root.
RUN chmod +x /entrypoint.sh && mkdir -p models data && chmod -R 777 models data
ENV HOME=/tmp

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s \
    CMD curl -fsS "http://localhost:${PORT:-8000}/health" || exit 1

ENTRYPOINT ["/entrypoint.sh"]
