#!/bin/sh
# Container entrypoint: fetch the local GGUF model if configured, then serve.
set -eu

MODEL_PATH="${MODEL_PATH:-models/gemma-3-1b-it-q4_0.gguf}"
PORT="${PORT:-8000}"

if [ -n "${MODEL_URL:-}" ] && [ ! -f "$MODEL_PATH" ]; then
    echo "[entrypoint] downloading local model to $MODEL_PATH ..."
    mkdir -p "$(dirname "$MODEL_PATH")"
    if command -v curl >/dev/null 2>&1; then
        curl -fSL --retry 3 -o "$MODEL_PATH.tmp" "$MODEL_URL" && mv "$MODEL_PATH.tmp" "$MODEL_PATH"
    else
        echo "[entrypoint] curl not available; skipping model download" >&2
    fi
fi

if [ ! -f "$MODEL_PATH" ]; then
    echo "[entrypoint] no local model at $MODEL_PATH - starting in remote-only mode"
fi

exec python -m uvicorn routing_agent.web.app:create_default_app \
    --factory --host 0.0.0.0 --port "$PORT"
