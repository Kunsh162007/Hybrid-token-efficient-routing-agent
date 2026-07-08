---
title: Hybrid Token-Efficient Routing Agent
emoji: 🪜
colorFrom: green
colorTo: yellow
sdk: docker
app_port: 8000
pinned: false
---

# Hybrid Token-Efficient Routing Agent

**AMD Developer Hackathon: ACT II — Track 1.** An agent that completes tasks
with the fewest possible *paid* tokens: a local Gemma model (free under the
scoring rules) handles everything it can prove it can handle, and Fireworks AI
is called only when the free rungs cannot be trusted.

## The escalation ladder

Every task climbs a cost-ordered ladder and exits at the first trustworthy rung:

| Rung | Action | Cost |
|---|---|---|
| 0 | Heuristic/learned classify + cache lookup | free |
| 1 | Local Gemma attempt + verifier | free |
| 2 | Local retry (reworded, hotter) | free |
| 3 | Self-consistency majority vote (k samples) | free |
| 4 | Remote judge: 1-token YES/NO on the local winner | ~50-200 tokens |
| 5 | Cheap remote model, compressed prompt + draft hint | paid |
| 6 | Strong remote model | paid, last resort |

Extra machinery that keeps tokens down:

- **Logprob confidence** — `exp(mean token logprob)` from llama.cpp gates local
  answers honestly; no "are you sure?" prompts.
- **Adaptive thresholds** — per-task-type success EMA lowers/raises the local
  bar during a run. Zero tokens, no training.
- **Semantic answer cache** — paid answers are stored (SQLite + MiniLM
  embeddings); near-duplicate queries are free forever after.
- **Learned router** — logistic regression trained on eval-run records predicts
  P(local succeeds); enable after it beats the heuristics.
- **Task decomposer** (optional, kill-switched) — the local model splits big
  tasks so only irreducible subtasks pay.
- **Budget guardrails** — hard per-task remote-token cap; on exhaustion the
  agent ships its best free answer.

## Quickstart

```bash
git clone <this repo> && cd Hybrid-token-efficient-routing-agent
python -m venv .venv && . .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[all]"                           # or ".[web,dev]" without local inference

# Local model (Gemma 3 1B instruct, Q4_0, ~700MB)
huggingface-cli download google/gemma-3-1b-it-qat-q4_0-gguf \
    gemma-3-1b-it-q4_0.gguf --local-dir models/

cp .env.example .env                              # add your FIREWORKS_API_KEY
```

### Use it

```bash
routing-agent route "What is 128 * 46?"           # one task
routing-agent run  --tasks tasks/sample_tasks.jsonl
routing-agent eval --tasks tasks/sample_tasks.jsonl --train-log data/training_records.jsonl
routing-agent train-router --log data/training_records.jsonl --out data/router_model.joblib
routing-agent serve                               # dashboard at http://localhost:8000
```

Task files are JSONL: `{"id": "t1", "prompt": "...", "expected": "...", "task_type": "math"}`
(`task_type` optional — it is inferred).

### Docker

```bash
docker build -t routing-agent .
docker run -p 8000:8000 -e FIREWORKS_API_KEY=fw_... \
  -e MODEL_URL="https://huggingface.co/google/gemma-3-1b-it-qat-q4_0-gguf/resolve/main/gemma-3-1b-it-q4_0.gguf" \
  routing-agent
```

Without `MODEL_URL` (or without the RAM for it) the agent runs remote-only and
says so on `/health`. Deployment: the public image lives at
`docker.io/kunsh16/routing-agent:latest`; a live demo runs on Hugging Face
Spaces (Docker SDK — push this repo with the model baked via `MODEL_URL`).

### Harness / submission mode

The same image satisfies the hackathon judging contract: when
`/input/tasks.json` exists (or `HARNESS_MODE=1`), the container routes every
task, writes `/output/results.json`, and exits 0 — no web server.

```bash
docker run -v "$PWD/input:/input" -v "$PWD/output:/output" \
  -e FIREWORKS_API_KEY=... -e FIREWORKS_BASE_URL=... -e ALLOWED_MODELS=... \
  routing-agent
```

Compliance handled automatically:

- `FIREWORKS_BASE_URL` and `ALLOWED_MODELS` (injected by the harness)
  **override** `config.yaml` — cheap/strong tiers are picked from the allowed
  list by parameter-count hints in the model IDs.
- The answer cache is disabled in this mode (no cached answers rule).
- A global ~9-minute budget shrinks the per-task wall clock (≤25s/task) so the
  run always finishes inside the 10-minute limit; `results.json` is rewritten
  atomically after every task, so it is valid JSON at any kill point.
- For the fastest cold start, bake the GGUF into the image:
  `docker build --build-arg MODEL_URL=https://... -t routing-agent .`

## Launch-day checklist (models revealed at kickoff)

1. Nothing to hardcode: the harness injects `ALLOWED_MODELS` and
   `FIREWORKS_BASE_URL` at runtime. For local eval runs, optionally mirror the
   revealed IDs into `config.yaml` (`remote.*`) or export the env vars.
2. `routing-agent eval --tasks <revealed-or-proxy tasks> --train-log data/training_records.jsonl`
3. Sweep thresholds (`eval` + edit `ladder.confidence_threshold`) or train the
   learned router and set `learned_router.enabled: true` if it wins.
4. A/B the decomposer per task type; enable `decomposer.enabled` only if it
   measurably reduces tokens at equal accuracy.

## Configuration

Everything lives in [`config.yaml`](config.yaml) — model IDs, ladder
thresholds, budgets, cache, decomposer kill-switch, demo mode. The only
secret is `FIREWORKS_API_KEY` (environment / `.env`, never committed).

## Development

```bash
pip install -e ".[dev]"
pytest            # 108 tests, all offline (clients are mocked)
ruff check src tests
```

## License

MIT
