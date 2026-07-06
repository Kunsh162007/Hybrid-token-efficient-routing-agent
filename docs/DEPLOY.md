# Deployment

Two supported free targets:

| | Hugging Face Spaces (free) | Render (free) |
|---|---|---|
| RAM | 16GB — **local Gemma fits**, full ladder | 512MB — remote-only |
| Sleep | after ~48h inactivity | after ~15 min inactivity |
| Best for | the real demo (free rungs visible) | always-on health URL |

## Hugging Face Spaces (recommended — runs the complete project)

1. Create an account at huggingface.co, then **New Space** → SDK: **Docker** →
   Blank template → CPU basic (free) → public.
2. The repo's `README.md` front matter and `Dockerfile` are already
   Space-compatible (`app_port: 8000`, non-root-writable dirs).
3. Push this repo to the Space (add it as a second git remote):

   ```bash
   git remote add space https://huggingface.co/spaces/<username>/<space-name>
   git push space main
   ```

   Authenticate with a **write** token from hf.co/settings/tokens.
4. In the Space → Settings:
   - **Secret** `FIREWORKS_API_KEY` = your key
   - **Variable** `MODEL_URL` = `https://huggingface.co/ggml-org/gemma-3-1b-it-GGUF/resolve/main/gemma-3-1b-it-Q4_K_M.gguf`
5. First build ~10-15 min (llama.cpp compiles) + model download at start.
   `/health` should show `"local_model": true`.

# Deploying to Render

The repo ships a `render.yaml` Blueprint, so deployment is mostly clicking.
Free tier is remote-only (no RAM for the local model).

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
