"""Run a task set through the ladder and report accuracy vs. token spend.

Side effect: every run appends labeled records (did local succeed?) to the
training log, which is exactly the data the learned router trains on.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from routing_agent.eval.scorers import score
from routing_agent.router.classifier import classify
from routing_agent.types import Rung, TaskType


@dataclass(frozen=True)
class Task:
    id: str
    prompt: str
    expected: str
    task_type: TaskType


@dataclass(frozen=True)
class TaskRow:
    task_id: str
    correct: bool
    remote_tokens: int
    exit_rung: Rung
    cached: bool
    answer: str


@dataclass(frozen=True)
class EvalReport:
    rows: tuple[TaskRow, ...]
    accuracy: float
    total_remote_tokens: int
    free_task_ratio: float
    rung_exits: dict[Rung, int] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"tasks:          {len(self.rows)}",
            f"accuracy:       {self.accuracy:.1%}",
            f"remote tokens:  {self.total_remote_tokens}",
            f"free tasks:     {self.free_task_ratio:.1%}",
            "rung exits:",
        ]
        for rung in sorted(self.rung_exits):
            lines.append(f"  {rung.name:<16} {self.rung_exits[rung]}")
        return "\n".join(lines)


class TaskFormatError(ValueError):
    """A task line in the JSONL file is malformed."""


def load_tasks(path: str | Path) -> list[Task]:
    """Parse a JSONL task file, validating every line."""
    tasks: list[Task] = []
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise TaskFormatError(f"Line {line_number}: invalid JSON: {exc}") from exc
        if not isinstance(raw, dict) or "prompt" not in raw:
            raise TaskFormatError(f"Line {line_number}: needs at least a 'prompt' field")
        declared = raw.get("task_type")
        task_type = (
            TaskType(declared) if declared else classify(raw["prompt"]).task_type
        )
        tasks.append(
            Task(
                id=str(raw.get("id", line_number)),
                prompt=raw["prompt"],
                expected=str(raw.get("expected", "")),
                task_type=task_type,
            )
        )
    return tasks


def run_eval(
    ladder,
    tasks: list[Task],
    *,
    training_log_path: str | Path | None = None,
) -> EvalReport:
    """Route every task, score it, and aggregate the cost/accuracy report."""
    rows: list[TaskRow] = []
    rung_exits: dict[Rung, int] = {}
    training_records: list[dict] = []

    for task in tasks:
        result = ladder.route(task.prompt)
        correct = score(task.task_type, task.expected, result.answer) if task.expected else False
        rows.append(
            TaskRow(
                task_id=task.id,
                correct=correct,
                remote_tokens=result.remote_tokens,
                exit_rung=result.exit_rung,
                cached=result.cached,
                answer=result.answer,
            )
        )
        rung_exits[result.exit_rung] = rung_exits.get(result.exit_rung, 0) + 1
        # Label for the learned router: did the local ladder handle it correctly?
        local_success = correct and result.exit_rung <= Rung.REMOTE_JUDGE
        training_records.append(
            {"prompt": task.prompt, "task_type": str(task.task_type), "label": int(local_success)}
        )

    if training_log_path is not None:
        log_path = Path(training_log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            for record in training_records:
                handle.write(json.dumps(record) + "\n")

    correct_count = sum(1 for row in rows if row.correct)
    free_count = sum(1 for row in rows if row.remote_tokens == 0)
    return EvalReport(
        rows=tuple(rows),
        accuracy=correct_count / len(rows) if rows else 0.0,
        total_remote_tokens=sum(row.remote_tokens for row in rows),
        free_task_ratio=free_count / len(rows) if rows else 0.0,
        rung_exits=rung_exits,
    )
