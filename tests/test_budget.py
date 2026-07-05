"""Budget tracker tests."""

import pytest

from routing_agent.budget import BudgetExceeded, BudgetTracker
from routing_agent.types import GenerationResult, Rung


def _remote(tokens: int) -> GenerationResult:
    return GenerationResult(
        text="x", model_id="r", is_remote=True,
        prompt_tokens=tokens, completion_tokens=0,
    )


def _local(tokens: int) -> GenerationResult:
    return GenerationResult(
        text="x", model_id="l", is_remote=False,
        prompt_tokens=tokens, completion_tokens=0,
    )


def test_local_generations_never_count_toward_budget():
    tracker = BudgetTracker(per_task_budget=100)
    tracker.begin_task()
    tracker.record(_local(10_000))
    tracker.check_remaining(100)  # should not raise
    assert tracker.task_remote_tokens == 0


def test_budget_exceeded_blocks_next_remote_call():
    tracker = BudgetTracker(per_task_budget=100)
    tracker.begin_task()
    tracker.record(_remote(90))
    with pytest.raises(BudgetExceeded):
        tracker.check_remaining(20)


def test_end_task_returns_spend_and_tracks_free_ratio():
    tracker = BudgetTracker(per_task_budget=1000)

    tracker.begin_task()
    tracker.record(_local(500))
    assert tracker.end_task(Rung.LOCAL_FIRST) == 0

    tracker.begin_task()
    tracker.record(_remote(120))
    assert tracker.end_task(Rung.REMOTE_CHEAP) == 120

    stats = tracker.snapshot()
    assert stats.tasks_completed == 2
    assert stats.free_task_count == 1
    assert stats.free_task_ratio == 0.5
    assert stats.remote_tokens_spent == 120
    assert stats.local_tokens_used == 500
    assert stats.rung_exits == {Rung.LOCAL_FIRST: 1, Rung.REMOTE_CHEAP: 1}


def test_budget_resets_between_tasks():
    tracker = BudgetTracker(per_task_budget=100)
    tracker.begin_task()
    tracker.record(_remote(95))
    tracker.end_task(Rung.REMOTE_CHEAP)

    tracker.begin_task()
    tracker.check_remaining(50)  # fresh budget, should not raise
