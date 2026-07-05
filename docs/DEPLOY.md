# Deploying to Render

The repo ships a `render.yaml` Blueprint, so deployment is mostly clicking.

## One-time setup

1. Push this repository to GitHub (public, required for the hackathon anyway).
2. In [Render](https://dashboard.render.com): **New → Blueprint**, pick the repo.
3. Render reads `render.yaml` and creates the `routing-agent` web service.
4. Set the one secret it asks for: `FIREWORKS_API_KEY`.
5. Deploy. First build takes ~10-15 min (llama-cpp-python compiles from source).

## Free tier vs. Standard

| | Free (512MB) | Standard (2GB, ~$25/mo prorated) |
|---|---|---|
| Local Gemma 3 1B | does **not** fit — remove `MODEL_URL` env var to skip it | fits, demo shows real local routing |
| Behavior | remote-only mode (dashboard still works; every task shows paid rungs) | full ladder, free rungs visible |
| Cold starts | spins down after ~15 min idle; first hit takes ~1 min | always on |

Recommendation: deploy free first to verify, upgrade for the demo video, then
downgrade after recording.

## Demo mode

Set `demo_mode: true` under `web:` in `config.yaml` (or bake a demo config)
to cap generation lengths so the dashboard feels snappy on CPU.

## Before judging / recording

- Hit the URL once to warm the instance (cold start + model download).
- `GET /health` should report `"local_model": true` on Standard.
- Run the four sample chips in the dashboard once so run totals look alive.
