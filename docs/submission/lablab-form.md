# lablab.ai Submission Form Copy

Field limits (from the live form): title 5-50 chars, short description
50-255 chars, long description 600-2000 chars (min 100 words).

## Submission Title (47/50 chars)

Hybrid Routing Agent — Local First, Tokens Last

## Short Description (192/255 chars)

A local Gemma model answers everything it can prove correct for free;
Fireworks AI is called only when trust runs out — starting with a 1-token
judge. Fewer paid tokens, same accuracy.

## Long Description (~1,850/2000 chars)

Track 1 scores accuracy first, then ranks survivors by Fireworks tokens spent
— and local tokens count as zero. So the winning agent isn't the one with the
best model; it's the one that most reliably knows when its free answer is
already correct.

Every task climbs a cost-ordered escalation ladder and exits at the first
trustworthy rung. Rung 0 classifies the task with zero-cost heuristics, plus
an exact Python solver for explicit arithmetic (free AND certain). Rungs 1-3
are local Gemma attempts gated by a strict free verifier (numbers must parse,
MCQ letters must be among the options, Python must compile, summaries must
honor stated length limits) and honest logprob confidence, then
self-consistency voting across independent samples. Rung 4 is the cheapest
paid call: a remote judge grades the local winner with a single output token
— verification costs 10-50x less than generation. Only then come real paid
generations: a cheap model with a compressed prompt and the local draft as a
hint, then a strong model as last resort, under a hard per-task token budget.

Weak-verifier categories (math word problems, code debugging, logical
deduction) never ship on local confidence alone — the 1-token judge must
approve, and a judge-rejected answer can never win a self-consistency vote.
That discipline came from dry-running the judged contract end to end: all
eight capability categories answered correctly at a fraction of all-remote
token cost.

It ships as a single public linux/amd64 Docker image implementing the harness
contract — tasks.json in, results.json out, atomic writes after every task,
10-minute budget management — reading FIREWORKS_BASE_URL and ALLOWED_MODELS
from the environment at runtime. Plus a live dashboard that visualizes every
task climbing the ladder with a token meter, 161 offline tests, and graceful
degradation when the model file or API key is absent.

## Technologies Used

Gemma 3 · Fireworks AI · llama.cpp · Python · FastAPI · SQLite · fastembed ·
scikit-learn · Docker · Hugging Face Spaces

Tag-style: `gemma` `fireworks-ai` `llama-cpp` `python` `fastapi` `docker`
`ai-agents` `model-routing` `cost-optimization` `amd`

## Links

- Public GitHub repository: https://github.com/Kunsh162007/Hybrid-token-efficient-routing-agent
- Docker image (Track 1 submission): docker.io/kunsh16/routing-agent:latest
- Live demo (HF Spaces): https://huggingface.co/spaces/Kunsh16/routing-agent
- Video presentation: [ADD URL after recording]
- Slides: print `docs/submission/slides.html` to PDF and upload
- Cover image: open `docs/submission/cover-image.html` at 1280x720, screenshot
