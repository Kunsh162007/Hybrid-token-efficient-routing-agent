"""Harness submission contract tests: tasks.json in, results.json out."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from routing_agent.submission import run_submission


@dataclass
class _StubResult:
    answer: str


@dataclass
class _StubRuntime:
    """Minimal route_task seam; scripted answers keyed by prompt."""

    answers: dict[str, str] = field(default_factory=dict)
    fail_prompts: set[str] = field(default_factory=set)
    calls: list[dict] = field(default_factory=list)

    def route_task(self, prompt, *, time_cap_seconds=None):
        self.calls.append({"prompt": prompt, "time_cap_seconds": time_cap_seconds})
        if prompt in self.fail_prompts:
            raise RuntimeError("scripted routing failure")
        return _StubResult(answer=self.answers.get(prompt, "fallback"))


def _write_tasks(tmp_path, tasks):
    path = tmp_path / "tasks.json"
    path.write_text(json.dumps(tasks), encoding="utf-8")
    return path


def _read_results(tmp_path):
    return json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))


def test_submission_writes_answer_for_every_task(tmp_path):
    # Arrange
    tasks = _write_tasks(
        tmp_path,
        [
            {"task_id": "t1", "prompt": "What is 2+2?"},
            {"task_id": "t2", "prompt": "Summarise cats."},
        ],
    )
    runtime = _StubRuntime(answers={"What is 2+2?": "4", "Summarise cats.": "Cats."})

    # Act
    code = run_submission(tasks, tmp_path / "results.json", runtime=runtime)

    # Assert
    assert code == 0
    assert _read_results(tmp_path) == [
        {"task_id": "t1", "answer": "4"},
        {"task_id": "t2", "answer": "Cats."},
    ]


def test_submission_task_failure_ships_empty_answer_not_crash(tmp_path):
    tasks = _write_tasks(
        tmp_path,
        [
            {"task_id": "bad", "prompt": "explodes"},
            {"task_id": "good", "prompt": "fine"},
        ],
    )
    runtime = _StubRuntime(answers={"fine": "ok"}, fail_prompts={"explodes"})

    code = run_submission(tasks, tmp_path / "results.json", runtime=runtime)

    assert code == 0
    results = {row["task_id"]: row["answer"] for row in _read_results(tmp_path)}
    assert results == {"bad": "", "good": "ok"}


def test_submission_exhausted_budget_still_writes_valid_json(tmp_path):
    tasks = _write_tasks(tmp_path, [{"task_id": "t1", "prompt": "anything"}])
    runtime = _StubRuntime()

    code = run_submission(
        tasks, tmp_path / "results.json", runtime=runtime, time_budget_seconds=0
    )

    assert code == 0
    assert _read_results(tmp_path) == [{"task_id": "t1", "answer": ""}]
    assert runtime.calls == []  # never routed: budget already spent


def test_submission_per_task_cap_stays_under_request_limit(tmp_path):
    tasks = _write_tasks(
        tmp_path, [{"task_id": f"t{i}", "prompt": f"q{i}"} for i in range(3)]
    )
    runtime = _StubRuntime()

    run_submission(tasks, tmp_path / "results.json", runtime=runtime)

    assert all(call["time_cap_seconds"] <= 25.0 for call in runtime.calls)


def test_submission_unwritable_output_fails_fast_with_nonzero_exit(tmp_path):
    tasks = _write_tasks(tmp_path, [{"task_id": "t1", "prompt": "q"}])
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a directory", encoding="utf-8")
    runtime = _StubRuntime()

    # Output parent is a file: mkdir fails, so no results can ever be written.
    code = run_submission(tasks, blocker / "results.json", runtime=runtime)

    assert code == 1
    assert runtime.calls == []  # fail fast: no time burned routing tasks


def test_submission_harness_config_disables_cache_and_bounds_remote():
    from routing_agent.config import AppConfig
    from routing_agent.submission import _harness_config

    cfg = _harness_config(AppConfig())

    assert cfg.cache.enabled is False
    assert cfg.decomposer.enabled is False
    assert cfg.ladder.wall_clock_cap_seconds <= 25.0
    assert cfg.remote.timeout_seconds <= 12.0
    assert cfg.remote.max_retries <= 1


def test_submission_missing_input_returns_nonzero_with_empty_results(tmp_path):
    code = run_submission(
        tmp_path / "missing.json", tmp_path / "results.json", runtime=_StubRuntime()
    )

    assert code == 1
    assert _read_results(tmp_path) == []


def test_submission_tolerates_missing_ids_and_empty_prompts(tmp_path):
    tasks = _write_tasks(tmp_path, [{"prompt": "hello"}, {"task_id": "t2"}])
    runtime = _StubRuntime(answers={"hello": "hi"})

    code = run_submission(tasks, tmp_path / "results.json", runtime=runtime)

    assert code == 0
    results = _read_results(tmp_path)
    assert results[0] == {"task_id": "0", "answer": "hi"}
    assert results[1] == {"task_id": "t2", "answer": ""}
