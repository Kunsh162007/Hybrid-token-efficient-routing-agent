# Final Submission Checklist

## Code & repo
- [ ] Push to a **public** GitHub repository
- [ ] README setup instructions verified on a clean machine (`pip install -e ".[all]"` → `routing-agent serve`)
- [ ] `docker build -t routing-agent .` succeeds; `docker run` serves `/health`
- [ ] `pytest` green; no secrets anywhere (`git log -p | grep -i fw_` clean)
- [ ] `.env` NOT committed (already gitignored — verify)

## Launch day (models revealed)
- [ ] Update `config.yaml`: `remote.cheap_model`, `remote.strong_model`, `remote.judge_model`, local GGUF
- [ ] Run `routing-agent eval` on revealed/proxy tasks with `--train-log`
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
