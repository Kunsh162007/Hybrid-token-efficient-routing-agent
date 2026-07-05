"""Threshold sweeps: find the cheapest config that stays above the accuracy bar."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from routing_agent.eval.harness import EvalReport, Task, run_eval


@dataclass(frozen=True)
class SweepPoint:
    confidence_threshold: float
    accuracy: float
    remote_tokens: int
    free_task_ratio: float


def sweep_confidence_thresholds(
    ladder_factory: Callable[[float], object],
    tasks: list[Task],
    thresholds: list[float],
) -> list[SweepPoint]:
    """Evaluate the task set once per candidate threshold.

    ladder_factory must build a *fresh* ladder (and budget) per threshold so
    runs do not share adaptive state.
    """
    points: list[SweepPoint] = []
    for threshold in thresholds:
        ladder = ladder_factory(threshold)
        report: EvalReport = run_eval(ladder, tasks)
        points.append(
            SweepPoint(
                confidence_threshold=threshold,
                accuracy=report.accuracy,
                remote_tokens=report.total_remote_tokens,
                free_task_ratio=report.free_task_ratio,
            )
        )
    return points


def cheapest_above(points: list[SweepPoint], accuracy_floor: float) -> SweepPoint | None:
    """The sweep point spending the fewest tokens while meeting the floor."""
    eligible = [p for p in points if p.accuracy >= accuracy_floor]
    if not eligible:
        return None
    return min(eligible, key=lambda p: p.remote_tokens)
