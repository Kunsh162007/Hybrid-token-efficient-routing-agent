# Video Script (2:30 target)

Record the Render dashboard (warmed up!) + one terminal. 1080p, cursor visible.

## Shot list

**[0:00-0:20] Hook — slide 1 or cover image on screen**
> "In this hackathon track, every remote token costs you leaderboard points —
> but local tokens are free. So we didn't build a smarter model. We built an
> agent that knows when its free answer is already right."

**[0:20-0:50] The ladder — dashboard, ladder panel visible**
> "Every task climbs this escalation ladder. Rungs zero to three run on a local
> Gemma model — classify, attempt, verify, retry, then a self-consistency vote.
> All free. Only when the free rungs can't agree do we pay: first a one-token
> remote judge, then a cheap Fireworks model, and only as a last resort the
> strong one."

*Action: click the `math` sample chip → Route it. Rung 1 lights green.*
> "Simple math: answered locally, verified by logprob confidence — zero tokens."

**[0:50-1:20] Contested case**
*Action: paste a harder task (prepare one that escalates).*
> "Here the local samples disagree — watch. The vote is contested, so instead of
> paying for a full remote generation, we send the local winner to a remote
> judge that answers with literally one token: yes or no. Verification is ten to
> fifty times cheaper than generation."

**[1:20-1:45] Run totals + cache**
*Action: point at stats; re-run the same hard task → CACHE HIT.*
> "The dashboard tracks the whole run: remote tokens, free-task ratio, cache
> hits. And every paid answer is cached semantically — ask a near-duplicate
> question and it's free forever after."

**[1:45-2:10] Launch-day readiness — terminal**
*Action: run `routing-agent eval --tasks tasks/sample_tasks.jsonl`.*
> "Models are revealed at kickoff, so every model ID lives in one config file.
> The eval harness measures accuracy against tokens, sweeps thresholds, and even
> generates training data for a learned router — we retune in minutes, not hours."

**[2:10-2:30] Close — slide 8**
> "Gemma running locally on llama.cpp, Fireworks AI for escalation, fully
> containerized, deployed on Render. Hybrid Token-Efficient Routing Agent:
> local first, tokens last."

## Prep checklist

- [ ] Warm the Render instance 10 min before recording
- [ ] Pre-test the "hard task" so it reliably escalates on camera
- [ ] Run 3-4 tasks beforehand so run totals are non-zero
- [ ] Close other tabs; hide bookmarks bar
