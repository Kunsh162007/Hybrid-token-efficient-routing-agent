"""Adaptive per-task-type escalation thresholds.

Tracks an exponential moving average of local success per task type during a
run. Types the local model keeps getting right need less confidence to ship
locally; types that keep escalating need more. Zero tokens, no training.
"""

from __future__ import annotations

import threading

from routing_agent.types import TaskType

_EMA_ALPHA = 0.3
_INITIAL_SUCCESS_RATE = 0.5
_MAX_ADJUSTMENT = 0.15  # threshold never moves more than this from base


class AdaptiveThresholds:
    """Thread-safe online threshold table keyed by task type."""

    def __init__(self, base_threshold: float) -> None:
        self._base = base_threshold
        self._lock = threading.Lock()
        self._ema: dict[TaskType, float] = {}

    def get(self, task_type: TaskType) -> float:
        """Confidence needed to ship a local answer for this task type."""
        with self._lock:
            rate = self._ema.get(task_type, _INITIAL_SUCCESS_RATE)
        # High local success -> lower bar; frequent escalation -> higher bar.
        adjustment = (rate - _INITIAL_SUCCESS_RATE) * 2 * _MAX_ADJUSTMENT
        return _clamp(self._base - adjustment, 0.05, 0.99)

    def update(self, task_type: TaskType, success: bool) -> None:
        with self._lock:
            previous = self._ema.get(task_type, _INITIAL_SUCCESS_RATE)
            observation = 1.0 if success else 0.0
            self._ema[task_type] = previous + _EMA_ALPHA * (observation - previous)

    def snapshot(self) -> dict[TaskType, float]:
        with self._lock:
            return dict(self._ema)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
