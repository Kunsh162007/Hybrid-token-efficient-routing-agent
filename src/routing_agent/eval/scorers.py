"""Answer scoring against expected values, per task type."""

from __future__ import annotations

from routing_agent.router.verifier import normalize
from routing_agent.types import TaskType

_NUMERIC_TOLERANCE = 1e-6


def score(task_type: TaskType, expected: str, answer: str) -> bool:
    """True when the answer matches the expected value after normalization."""
    got = normalize(task_type, answer)
    want = normalize(task_type, expected)

    if task_type == TaskType.MATH:
        return _numbers_match(want, got)
    if task_type == TaskType.MCQ:
        return got == want
    if task_type == TaskType.CODE:
        return _collapse(want) == _collapse(got) or want in got
    # Free-text: exact normalized match, or the expected value contained.
    return got == want or (len(want) >= 3 and want in got)


def _numbers_match(want: str, got: str) -> bool:
    try:
        return abs(float(want) - float(got)) <= _NUMERIC_TOLERANCE
    except ValueError:
        return want == got


def _collapse(text: str) -> str:
    return "".join(text.split())
