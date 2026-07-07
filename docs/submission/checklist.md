# Final Submission Checklist

## Harness contract (Track 1 — this is what gets scored)
- [ ] Image pushed to a **public registry** (GHCR/Docker Hub), `linux/amd64` manifest
- [ ] Compressed image size < 10 GB
- [ ] Dry run: `docker run -v .../input:/input -v .../output:/output -e FIREWORKS_API_KEY=... -e FIREWORKS_BASE_URL=... -e ALLOWED_MODELS=... routing-agent`
      reads `/input/tasks.json`, writes valid `/output/results.json`, exits 0
- [ ] Full dry run finishes well under 10 minutes; container ready < 60s
      (bake the GGUF: `--build-arg MODEL_URL=...`)
- [ ] No `.env` in the image (`.dockerignore` covers it) — env vars come from the harness
- [ ] No hardcoded model IDs in code; `ALLOWED_MODELS` override verified

## Code & repo
- [ ] Push to a **public** GitHub repository
- [ ] README setup instructions verified on a clean machine (`pip install -e ".[all]"` → `routing-agent serve`)
- [ ] `docker build -t routing-agent .` succeeds; `docker run` serves `/health`
- [ ] `pytest` green; no secrets anywhere (`git log -p | grep -i fw_` clean)
- [ ] `.env` NOT committed (already gitignored — verify)

## Launch day (models revealed)
- [ ] Verify `ALLOWED_MODELS` tier-picking on the real ID list (`load_config` test with the actual string)
- [ ] A/B `logits_all=False` in `clients/local.py` on the container's llama-cpp-python version
      (big prefill/memory win IF sampled-token logprobs still come through; some versions raise)
- [ ] Run `routing-agent eval` on revealed/proxy tasks with `--train-log`
- [ ] Run eval at least once with `task_type` omitted from every task — the real harness
      never supplies types, so classifier misrouting must show up in eval numbers
- [ ] Threshold sweep → pick cheapest point above the accuracy floor
- [ ] Train + A/B the learned router; enable only if it beats heuristics
- [ ] A/B the decomposer; keep `enabled: false` unless it wins

## Render demo
- [ ] Blueprint deployed, `FIREWORKS_API_KEY` set in dashboard
- [ ] Decide tier: free (remote-only) vs Standard (local Gemma live) for the video
- [ ] `/health` shows expected `local_model` value
- [ ] Warm instance + seed a few tasks before judging

## Assets
- [ ] Cover image: open `cover-image.html` at 1280x720, screenshot, upload
- [ ] Slides: open `slides.html`, print to PDF, upload
- [ ] Video: record per `video-script.md` (2-3 min), upload
- [ ] Form copy: paste from `lablab-form.md`, fill in the three URLs

## lablab.ai form fields
- [ ] Title, short description, long description
- [ ] Technology and category tags
- [ ] Cover image, video, slides
- [ ] GitHub URL, demo platform, application URL
- [ ] Submitted before the deadline (check Event Schedule tab!)
