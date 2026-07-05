# lablab.ai Submission Form Copy

## Project title

Hybrid Token-Efficient Routing Agent — Local First, Tokens Last

## Short description (elevator pitch)

An AI agent that treats remote tokens as the scarce resource they are: a local
Gemma model climbs a free escalation ladder (attempt → verify → self-consistency
vote), and Fireworks AI is called only when the free rungs can't be trusted —
starting with a 1-token remote judge before any paid generation. Semantic
caching, adaptive thresholds, and a learned router make it cheaper the longer
it runs.

## Long description

**The insight.** Track 1 scores remote token count plus output accuracy, and
local tokens count as zero. The winning agent therefore isn't the one with the
best model — it's the one that most reliably knows when its free answer is
already correct.

**The escalation ladder.** Every task climbs a cost-ordered ladder and exits at
the first trustworthy rung: (0) heuristic/learned classification and a semantic
cache lookup; (1-2) local Gemma attempts gated by a zero-cost verifier (numbers
must parse, MCQ letters must be among the options, Python must compile) and
logprob-based confidence — exp(mean token logprob), an honest signal instead of
asking the model if it's sure; (3) self-consistency majority voting across k
free samples, where a unanimous vote ships instantly; (4) the cheapest paid
rung — a remote judge that grades the local winner with literally one output
token, because verification costs 10-50x less than generation; (5) a cheap
Fireworks model with a compressed prompt and the local draft as a hint; (6) a
strong model as the last resort, under a hard per-task token budget that ships
the best free answer rather than overspending.

**It gets cheaper as it runs.** Paid answers are cached in SQLite with MiniLM
embeddings, so near-duplicate queries become free; per-task-type success rates
adapt the escalation thresholds online; and the eval harness turns every run
into training data for a logistic-regression router that predicts whether the
local model will succeed — retrainable in seconds when the real tasks and
models are revealed at kickoff (all model IDs live in one config file).

**Built like a product.** FastAPI dashboard that visualizes each task climbing
the ladder with a live token meter; CLI for scoring runs and evals; ~110
offline tests; containerized (CPU-only python:3.11-slim) so it runs identically
on a laptop, on Render, and on the scoring environment; graceful degradation
when the model file or the API key is absent.

**Stack.** Gemma 3 (local via llama.cpp, free) · Fireworks AI (escalation) ·
FastAPI · SQLite + fastembed · scikit-learn · Docker · Render.

## Technology & category tags

`gemma` `fireworks-ai` `llama.cpp` `python` `fastapi` `docker` `render`
`ai-agents` `model-routing` `cost-optimization` `amd`

## Links

- Public GitHub repository: [ADD URL]
- Demo application (Render): [ADD URL]
- Video presentation: [ADD URL]
- Slides: print `docs/submission/slides.html` to PDF and upload
