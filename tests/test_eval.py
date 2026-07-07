"""Eval harness, scorer, and sweep tests."""

import json

import pytest
from conftest import FakeLocalClient, FakeRemoteClient

from routing_agent.budget import BudgetTracker
from routing_agent.config import LadderConfig, LocalModelConfig, RemoteModelConfig
from routing_agent.eval.harness import TaskFormatError, load_tasks, run_eval
from routing_agent.eval.scorers import score
from routing_agent.eval.sweep import cheapest_above, sweep_confidence_thresholds
from routing_agent.router.ladder import EscalationLadder
from routing_agent.types import Rung, TaskType


def make_ladder(local, remote, threshold=0.72):
    return EscalationLadder(
        LadderConfig(confidence_threshold=threshold),
        LocalModelConfig(),
        RemoteModelConfig(),
        local,
        remote,
        BudgetTracker(per_task_budget=2000),
    )


# ------------------------------------------------------------------- scorers

def test_score_math_tolerates_formatting():
    assert score(TaskType.MATH, "5888", "The result is Answer: 5,888")


def test_score_math_rejects_wrong_number():
    assert not score(TaskType.MATH, "5888", "Answer: 5889")


def test_score_mcq_letter():
    assert score(TaskType.MCQ, "B", "Answer: (b)")


def test_score_qa_containment():
    assert score(TaskType.QA, "tokyo", "The capital is Tokyo.")


def test_score_code_whitespace_insensitive():
    assert score(
        TaskType.CODE,
        "def add(a, b):\n    return a + b",
        "```python\ndef add(a, b):\n    return a + b\n```",
    )


# ---------------------------------------------------------------- load_tasks

def test_load_tasks_parses_and_infers_type(tmp_path):
    path = tmp_path / "tasks.jsonl"
    path.write_text(
        '{"id": "t1", "prompt": "What is 2+2?", "expected": "4"}\n'
        '\n'
        '{"prompt": "Pick:\\nA) x\\nB) y", "expected": "A", "task_type": "mcq"}\n',
        encoding="utf-8",
    )
    tasks = load_tasks(path)
    assert len(tasks) == 2
    assert tasks[0].task_type == TaskType.MATH  # inferred
    assert tasks[1].task_type == TaskType.MCQ  # declared
    assert tasks[1].id == "3"  # falls back to line number


def test_load_tasks_rejects_bad_json(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(TaskFormatError, match="invalid JSON"):
        load_tasks(path)


def test_load_tasks_requires_prompt(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"id": "x"}\n', encoding="utf-8")
    with pytest.raises(TaskFormatError, match="prompt"):
        load_tasks(path)


# ------------------------------------------------------------------ run_eval

def test_run_eval_reports_accuracy_tokens_and_rungs(tmp_path):
    # QA tasks: not judge-gated, so confident local answers stay free.
    tasks_path = tmp_path / "tasks.jsonl"
    tasks_path.write_text(
        '{"id": "t1", "prompt": "Capital of France?", "expected": "paris", "task_type": "qa"}\n'
        '{"id": "t2", "prompt": "Capital of Germany?", "expected": "berlin", "task_type": "qa"}\n',
        encoding="utf-8",
    )
    tasks = load_tasks(tasks_path)
    # Local always answers Paris confidently: right for t1, wrong for t2.
    ladder = make_ladder(FakeLocalClient(answers=["Answer: Paris"], logprob_mean=-0.05),
                         FakeRemoteClient())

    report = run_eval(ladder, tasks)

    assert report.accuracy == 0.5
    assert report.total_remote_tokens == 0
    assert report.free_task_ratio == 1.0
    assert report.rung_exits == {Rung.LOCAL_FIRST: 2}
    assert "accuracy" in report.summary()


def test_run_eval_writes_training_log(tmp_path):
    tasks_path = tmp_path / "tasks.jsonl"
    tasks_path.write_text(
        '{"id": "t1", "prompt": "What is 2+2?", "expected": "4", "task_type": "math"}\n',
        encoding="utf-8",
    )
    log_path = tmp_path / "log" / "records.jsonl"
    ladder = make_ladder(FakeLocalClient(answers=["Answer: 4"], logprob_mean=-0.05),
                         FakeRemoteClient())

    run_eval(ladder, load_tasks(tasks_path), training_log_path=log_path)

    records = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert records == [{"prompt": "What is 2+2?", "task_type": "math", "label": 1}]


# --------------------------------------------------------------------- sweep

def test_sweep_and_cheapest_above(tmp_path):
    tasks_path = tmp_path / "tasks.jsonl"
    tasks_path.write_text(
        '{"id": "t1", "prompt": "Capital of France?", "expected": "paris", "task_type": "qa"}\n',
        encoding="utf-8",
    )
    tasks = load_tasks(tasks_path)

    def factory(threshold):
        return make_ladder(
            FakeLocalClient(answers=["Answer: Paris"], logprob_mean=-0.05),
            FakeRemoteClient(),
            threshold=threshold,
        )

    points = sweep_confidence_thresholds(factory, tasks, [0.5, 0.9])
    assert len(points) == 2
    assert all(p.accuracy == 1.0 for p in points)

    best = cheapest_above(points, accuracy_floor=0.99)
    assert best is not None and best.remote_tokens == 0
    assert cheapest_above(points, accuracy_floor=1.01) is None
