"""Hackathon harness batch mode: /input/tasks.json -> /output/results.json.

Contract (AMD Developer Hackathon participant guide, Track 1):
- read a JSON array of {"task_id", "prompt"} on startup
- write a JSON array of {"task_id", "answer"} before exiting
- exit 0 on success, non-zero on failure
- maximum runtime 10 minutes; per-request responses under 30 seconds

Robustness rules this module enforces:
- results.json is rewritten atomically after every task, so a hard kill at
  any point still leaves valid JSON covering every task_id
- a per-task exception never aborts the run; that task ships an empty answer
- the per-task wall-clock cap shrinks as the global deadline approaches
- the answer cache is disabled (guide: "do not hardcode or cache answers")
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING

from routing_agent.config import AppConfig, load_config

if TYPE_CHECKING:
    from routing_agent.runtime import Runtime

logger = logging.getLogger(__name__)

DEFAULT_INPUT_PATH = "/input/tasks.json"
DEFAULT_OUTPUT_PATH = "/output/results.json"
# Leave a safety margin under the harness's 10-minute kill.
DEFAULT_TIME_BUDGET_SECONDS = 540.0
# Stay under the guide's 30-seconds-per-request rule.
MAX_TASK_SECONDS = 25.0
# Below this cap the ladder cannot finish even one local attempt; escalation
# paths still respect it via the out-of-time checks.
MIN_TASK_SECONDS = 5.0
# Remote HTTP bounds for harness mode: the ladder only checks the wall clock
# *between* calls, so a single request must never be able to hold a task
# hostage. Worst case = 2 attempts x 12s + 1s backoff, inside the 30s rule.
REMOTE_TIMEOUT_SECONDS = 12.0
REMOTE_MAX_RETRIES = 1


def run_submission(
    input_path: str | Path = DEFAULT_INPUT_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    config_path: str | None = None,
    *,
    time_budget_seconds: float = DEFAULT_TIME_BUDGET_SECONDS,
    runtime=None,
) -> int:
    """Run the harness contract end to end; returns the process exit code.

    `runtime` is an injection seam for tests: anything with
    `route_task(prompt, time_cap_seconds=...)` returning an object with an
    `.answer` attribute.
    """
    started = time.monotonic()
    output = Path(output_path)

    try:
        tasks = _load_tasks(Path(input_path))
    except (OSError, ValueError) as exc:
        logger.error("Cannot read tasks from %s: %s", input_path, exc)
        _write_results(output, [])
        return 1

    # Every task_id gets an entry up front; routing only improves the answers.
    results = [{"task_id": task_id, "answer": ""} for task_id, _ in tasks]
    if not _write_results(output, results):
        # Unwritable output means a guaranteed zero score - fail fast instead
        # of burning the whole time budget first.
        logger.critical("Output path %s is not writable; aborting", output)
        return 1

    if runtime is None:
        runtime = _build_harness_runtime(config_path)
    if runtime is None:
        return 1

    for index, (task_id, prompt) in enumerate(tasks):
        if not prompt:
            continue  # already recorded as an empty answer
        remaining = time_budget_seconds - (time.monotonic() - started)
        tasks_left = len(tasks) - index
        if remaining <= 0:
            logger.warning("Global time budget exhausted; %d tasks unanswered", tasks_left)
            break
        cap = max(MIN_TASK_SECONDS, min(MAX_TASK_SECONDS, remaining / tasks_left))
        try:
            result = runtime.route_task(prompt, time_cap_seconds=cap)
            results[index] = {"task_id": task_id, "answer": result.answer}
        except Exception:
            logger.exception("Task %s failed; shipping empty answer", task_id)
        _write_results(output, results)

    if not _write_results(output, results):
        logger.critical("Final results write to %s failed", output)
        return 1

    answered = sum(1 for row in results if row["answer"])
    if tasks and answered == 0:
        # Valid JSON with all-empty answers scores zero on the accuracy gate;
        # make the systemic cause (auth, models, network) loud in the logs.
        logger.critical(
            "Every task produced an empty answer - check FIREWORKS_API_KEY, "
            "FIREWORKS_BASE_URL, ALLOWED_MODELS and the model file"
        )
    stats = getattr(runtime, "budget", None)
    if stats is not None:
        logger.info("Run stats: %s", stats.snapshot())
    logger.info(
        "Submission complete: %d/%d tasks answered in %.1fs",
        answered, len(tasks), time.monotonic() - started,
    )
    return 0


def _load_tasks(path: Path) -> list[tuple[str, str]]:
    """Parse the harness task file into (task_id, prompt) pairs."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"tasks.json root must be a list, got {type(raw).__name__}")
    tasks: list[tuple[str, str]] = []
    for position, entry in enumerate(raw):
        if not isinstance(entry, dict):
            logger.warning("Skipping non-object task at index %d", position)
            continue
        task_id = str(entry.get("task_id", position))
        prompt = entry.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            logger.warning("Task %s has no usable prompt; will answer empty", task_id)
            prompt = ""
        tasks.append((task_id, prompt))
    return tasks


def _build_harness_runtime(config_path: str | None) -> Runtime | None:
    """Build the runtime with harness-mode overrides, or None on failure."""
    from routing_agent.runtime import build_runtime

    try:
        config = _harness_config(load_config(config_path))
        return build_runtime(config=config)
    except Exception:
        logger.exception("Runtime construction failed")
        return None


def _harness_config(config: AppConfig) -> AppConfig:
    """Submission-mode overrides on top of the loaded config.

    The cache is disabled because the guide forbids cached answers; the
    per-task wall clock is pre-capped (route() shrinks it further per task);
    the decomposer is forced off because its subtasks each get a fresh time
    cap, which could multiply one task's share of the global budget.
    """
    return config.model_copy(
        update={
            "cache": config.cache.model_copy(update={"enabled": False}),
            "ladder": config.ladder.model_copy(
                update={"wall_clock_cap_seconds": MAX_TASK_SECONDS}
            ),
            "decomposer": config.decomposer.model_copy(update={"enabled": False}),
            "remote": config.remote.model_copy(
                update={
                    "timeout_seconds": min(
                        config.remote.timeout_seconds, REMOTE_TIMEOUT_SECONDS
                    ),
                    "max_retries": min(config.remote.max_retries, REMOTE_MAX_RETRIES),
                }
            ),
        }
    )


def _write_results(path: Path, results: list[dict]) -> bool:
    """Atomic write: the harness must never observe a half-written file.

    Returns False instead of raising so per-task loop writes stay best-effort;
    the caller decides which writes are fatal. ValueError covers surrogate
    code points a model could emit that json/utf-8 cannot encode.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(results, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)
        return True
    except (OSError, ValueError):
        logger.exception("Failed writing results to %s", path)
        return False
