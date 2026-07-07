"""Core immutable data types shared across the routing agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum, StrEnum


class TaskType(StrEnum):
    MATH = "math"
    MCQ = "mcq"
    CODE = "code"
    EXTRACTION = "extraction"
    SUMMARY = "summary"
    SENTIMENT = "sentiment"
    LOGIC = "logic"
    QA = "qa"
    GENERAL = "general"


class Rung(IntEnum):
    """Escalation ladder rungs, ordered by token cost."""

    CLASSIFY = 0
    LOCAL_FIRST = 1
    LOCAL_RETRY = 2
    SELF_CONSISTENCY = 3
    REMOTE_JUDGE = 4
    REMOTE_CHEAP = 5
    REMOTE_STRONG = 6


@dataclass(frozen=True)
class Classification:
    """Rung 0 output: what kind of task and how hard it looks."""

    task_type: TaskType
    difficulty: float  # 0.0 (trivial) .. 1.0 (clearly beyond the local model)
    signals: tuple[str, ...] = ()


@dataclass(frozen=True)
class GenerationResult:
    """One model generation, local or remote."""

    text: str
    model_id: str
    is_remote: bool
    prompt_tokens: int = 0
    completion_tokens: int = 0
    logprob_mean: float | None = None  # only local backends expose this

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def billed_tokens(self) -> int:
        """Tokens that count toward the score: local is always free."""
        return self.total_tokens if self.is_remote else 0


@dataclass(frozen=True)
class RungTrace:
    """One rung visited while routing a task."""

    rung: Rung
    action: str
    detail: str = ""
    remote_tokens: int = 0


@dataclass(frozen=True)
class TaskResult:
    """Final outcome of routing one task through the ladder."""

    answer: str
    exit_rung: Rung
    confidence: float
    remote_tokens: int
    task_type: TaskType
    cached: bool = False
    verified: bool = False
    elapsed_seconds: float = 0.0
    trace: tuple[RungTrace, ...] = field(default_factory=tuple)

    @property
    def was_free(self) -> bool:
        return self.remote_tokens == 0


@dataclass(frozen=True)
class VerifyResult:
    """Verifier verdict on a candidate answer."""

    ok: bool
    normalized: str
    reason: str = ""
