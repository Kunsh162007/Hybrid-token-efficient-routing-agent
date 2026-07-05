"""Core type behavior tests."""

import dataclasses

import pytest

from routing_agent.types import GenerationResult, Rung, TaskResult, TaskType


def test_local_generation_is_never_billed():
    # Arrange
    result = GenerationResult(
        text="42", model_id="gemma-local", is_remote=False,
        prompt_tokens=100, completion_tokens=50,
    )

    # Assert
    assert result.total_tokens == 150
    assert result.billed_tokens == 0


def test_remote_generation_bills_all_tokens():
    result = GenerationResult(
        text="42", model_id="fireworks/x", is_remote=True,
        prompt_tokens=100, completion_tokens=50,
    )
    assert result.billed_tokens == 150


def test_task_result_is_immutable():
    result = TaskResult(
        answer="a", exit_rung=Rung.LOCAL_FIRST, confidence=0.9,
        remote_tokens=0, task_type=TaskType.QA,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.answer = "b"  # type: ignore[misc]


def test_rungs_are_ordered_by_cost():
    assert Rung.LOCAL_FIRST < Rung.SELF_CONSISTENCY < Rung.REMOTE_JUDGE < Rung.REMOTE_STRONG


def test_was_free_reflects_remote_tokens():
    free = TaskResult(
        answer="a", exit_rung=Rung.SELF_CONSISTENCY, confidence=0.8,
        remote_tokens=0, task_type=TaskType.MATH,
    )
    paid = dataclasses.replace(free, remote_tokens=10)
    assert free.was_free and not paid.was_free
