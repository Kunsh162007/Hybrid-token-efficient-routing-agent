#!/bin/sh
# Container entrypoint. Two modes:
#   harness: /input/tasks.json exists (or HARNESS_MODE=1) -> batch submit, exit
#   demo:    otherwise -> web dashboard
set -eu

MODEL_PATH="${MODEL_PATH:-models/gemma-3-1b-it-q4_0.gguf}"
PORT="${PORT:-8000}"

HARNESS=0
if [ -f /input/tasks.json ] || [ "${HARNESS_MODE:-0}" = "1" ]; then
    HARNESS=1
fi

# Bound the model fetch: this shell time is invisible to the Python-side task
# budget, so in harness mode it must stay small (bake the model into the image
# instead - see Dockerfile). Demo mode can afford a longer fetch.
if [ "$HARNESS" = "1" ]; then
    MODEL_DOWNLOAD_TIMEOUT="${MODEL_DOWNLOAD_TIMEOUT:-45}"
else
    MODEL_DOWNLOAD_TIMEOUT="${MODEL_DOWNLOAD_TIMEOUT:-180}"
fi

if [ -n "${MODEL_URL:-}" ] && [ ! -f "$MODEL_PATH" ]; then
    echo "[entrypoint] downloading local model to $MODEL_PATH ..."
    mkdir -p "$(dirname "$MODEL_PATH")"
    if curl -fSL --retry 2 --max-time "$MODEL_DOWNLOAD_TIMEOUT" \
            -o "$MODEL_PATH.tmp" "$MODEL_URL"; then
        mv "$MODEL_PATH.tmp" "$MODEL_PATH"
    else
        echo "[entrypoint] model download failed; continuing without it" >&2
        rm -f "$MODEL_PATH.tmp"
    fi
fi

if [ ! -f "$MODEL_PATH" ]; then
    echo "[entrypoint] no local model at $MODEL_PATH - remote-only mode"
fi

if [ "$HARNESS" = "1" ]; then
    echo "[entrypoint] harness mode: tasks.json -> results.json"
    exec python -m routing_agent.cli submit
fi

exec python -m uvicorn routing_agent.web.app:create_default_app \
    --factory --host 0.0.0.0 --port "$PORT"
