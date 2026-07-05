# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
.venv/Scripts/python -m pytest                    # full suite (~110 tests, all offline)
.venv/Scripts/python -m pytest tests/test_ladder.py -k contested   # single test
.venv/Scripts/python -m ruff check src tests      # lint
pip install -e ".[dev,web]"                       # minimal dev install (no llama.cpp build)
pip install -e ".[all]"                           # + local inference, cache embeddings, sklearn
.venv/Scripts/python -m routing_agent.cli serve   # dashboard at :8000
docker build -t routing-agent .                   # python:3.11-slim; llama.cpp compiles from source
```

Tests never hit the network: remote calls use `httpx.MockTransport`, model
clients are the scripted fakes in `tests/conftest.py` (`FakeLocalClient`,
`FakeRemoteClient`). Tests import those via `from conftest import ...` (tests/
is not a package). Heavy deps degrade: `test_learned.py` skips without sklearn,
`test_web.py` without fastapi.

## Architecture

The core idea: **local tokens are free, remote tokens are scored** (AMD
hackathon Track 1). Everything routes through the escalation ladder in
`src/routing_agent/router/ladder.py` — rungs 0-3 are free local attempts
(verify → retry → self-consistency vote), rung 4 is a 1-token remote judge on
the local winner, rungs 5-6 are real remote generations. The ladder exits at
the first rung whose answer can be trusted.

Wiring flow: `cli.py` / `web/app.py` → `runtime.py:build_runtime()` (the only
place clients, cache, thresholds, ladder, decomposer are assembled) →
`EscalationLadder.route()`. The runtime degrades gracefully: missing GGUF →
remote-only; missing `FIREWORKS_API_KEY` → local-only; both missing → ConfigError.

Key contracts to preserve:

- `types.py` dataclasses are frozen; `GenerationResult.billed_tokens` is 0 for
  local results — that invariant is what the whole scoring strategy rests on.
- `EscalationLadder` takes pluggable seams: `cache` (needs `lookup`/`put`),
  `difficulty_estimator` (returns `Classification`; the learned router plugs in
  via `LearnedRouter.as_estimator()`), and clients (anything with `generate()`,
  remote also `judge()`). `None` local or remote client is legal.
- `BudgetTracker.check_remaining()` must be called *before* every paid call;
  `BudgetExceeded` is caught in `_climb` and settles for the best free answer.
- The verifier (`router/verifier.py`) is the gatekeeper between free and paid:
  keep it strict but zero-cost. `normalize()` also feeds voting and eval scoring,
  so changes affect three places.
- Everything launch-day-dependent (model IDs, thresholds) lives in
  `config.yaml` — never hardcode model names in source.

Eval loop: `eval/harness.py:run_eval()` scores accuracy vs. remote tokens and
appends training records; `router/learned.py` trains a logistic regression from
them. `eval/sweep.py` finds the cheapest threshold above an accuracy floor.

## Gotchas

- `scripts/entrypoint.sh` must stay LF (`.gitattributes` enforces it) or the
  Linux container fails to start.
- Windows dev box runs Python 3.14; the Docker image pins 3.11 for
  llama-cpp-python wheel/build compatibility. Don't raise `requires-python`.
- `web/static/index.html` is served raw by FastAPI (no build step); keep it
  self-contained (inline CSS/JS).
