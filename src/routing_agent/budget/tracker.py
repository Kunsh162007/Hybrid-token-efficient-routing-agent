"""Token budget tracking: per-task hard caps and run-level statistics."""

from __future__ import annotations

import threading
from dataclasses import dataclass

from routing_agent.types import GenerationResult, Rung


class BudgetExceeded(Exception):
    """The per-task remote-token budget would be exceeded."""


@dataclass(frozen=True)
class RunStats:
    """Immutable snapshot of run-level accounting."""

    tasks_completed: int
    remote_tokens_spent: int
    local_tokens_used: int
    free_task_count: int
    rung_exits: dict[Rung, int]

    @property
    def free_task_ratio(self) -> float:
        if self.tasks_completed == 0:
            return 0.0
        return self.free_task_count / self.tasks_completed


class BudgetTracker:
    """Accounts every generation and enforces the per-task remote budget.

    Thread-safe so the web dashboard can read stats while tasks run.
    """

    def __init__(self, per_task_budget: int) -> None:
        self._per_task_budget = per_task_budget
        self._lock = threading.Lock()
        self._task_remote_tokens = 0
        self._run_remote_tokens = 0
        self._run_local_tokens = 0
        self._tasks_completed = 0
        self._free_tasks = 0
        self._rung_exits: dict[Rung, int] = {}

    def begin_task(self) -> None:
        with self._lock:
            self._task_remote_tokens = 0

    def record(self, result: GenerationResult) -> None:
        with self._lock:
            if result.is_remote:
                self._task_remote_tokens += result.total_tokens
                self._run_remote_tokens += result.total_tokens
            else:
                self._run_local_tokens += result.total_tokens

    def check_remaining(self, estimated_next_call: int = 0) -> None:
        """Raise if the task budget is exhausted (before making another paid call)."""
        with self._lock:
            if self._task_remote_tokens + estimated_next_call > self._per_task_budget:
                raise BudgetExceeded(
                    f"Task remote budget {self._per_task_budget} exhausted "
                    f"({self._task_remote_tokens} spent)"
                )

    def end_task(self, exit_rung: Rung) -> int:
        """Close out the task; returns remote tokens the task consumed."""
        with self._lock:
            spent = self._task_remote_tokens
            self._tasks_completed += 1
            if spent == 0:
                self._free_tasks += 1
            self._rung_exits[exit_rung] = self._rung_exits.get(exit_rung, 0) + 1
            self._task_remote_tokens = 0
            return spent

    @property
    def task_remote_tokens(self) -> int:
        with self._lock:
            return self._task_remote_tokens

    def snapshot(self) -> RunStats:
        with self._lock:
            return RunStats(
                tasks_completed=self._tasks_completed,
                remote_tokens_spent=self._run_remote_tokens,
                local_tokens_used=self._run_local_tokens,
                free_task_count=self._free_tasks,
                rung_exits=dict(self._rung_exits),
            )
