"""Task decomposition: split hard tasks so easy parts stay local.

The *local* model does the splitting and composing (both free); only
irreducible subtasks climb to paid rungs. Guardrails: depth 1 (subtasks are
never re-split), max 4 subtasks, and a config kill-switch, because a bad
decomposition can cost more than it saves.
"""

from __future__ import annotations

import json
import re

from routing_agent.clients.base import GenerationError
from routing_agent.config import DecomposerConfig
from routing_agent.router.classifier import classify
from routing_agent.types import Rung, TaskResult

_SPLIT_SYSTEM = (
    "You split tasks into independent subtasks. Reply with only a JSON array "
    "of subtask strings. If the task cannot be usefully split, reply []."
)
_COMPOSE_SYSTEM = (
    "You combine subtask answers into one final answer for the original task. "
    "Reply with only the final answer."
)
_JSON_ARRAY = re.compile(r"\[.*\]", re.DOTALL)
_MIN_PROMPT_WORDS = 30  # short tasks are never worth splitting


class Decomposer:
    """Local-model task splitter with hard guardrails."""

    def __init__(self, config: DecomposerConfig, local_client) -> None:
        self._config = config
        self._local = local_client

    @property
    def enabled(self) -> bool:
        return self._config.enabled and self._local is not None

    def split(self, prompt: str) -> list[str]:
        """Return subtasks, or [] when splitting is disabled or unhelpful."""
        if not self.enabled or len(prompt.split()) < _MIN_PROMPT_WORDS:
            return []
        try:
            result = self._local.generate(
                f"Task:\n{prompt}\n\nSplit into at most "
                f"{self._config.max_subtasks} independent subtasks.",
                system=_SPLIT_SYSTEM,
            )
        except GenerationError:
            return []
        subtasks = _parse_subtasks(result.text)
        if len(subtasks) < 2:  # a 1-item split is just overhead
            return []
        return subtasks[: self._config.max_subtasks]

    def compose(self, prompt: str, subtask_answers: list[tuple[str, str]]) -> str:
        """Merge subtask answers locally; falls back to concatenation."""
        parts = "\n".join(
            f"Subtask: {sub}\nAnswer: {ans}" for sub, ans in subtask_answers
        )
        try:
            result = self._local.generate(
                f"Original task:\n{prompt}\n\n{parts}\n\nFinal answer:",
                system=_COMPOSE_SYSTEM,
            )
            if result.text.strip():
                return result.text.strip()
        except GenerationError:
            pass
        return "\n".join(answer for _, answer in subtask_answers if answer)

    def route_decomposed(self, prompt: str, ladder) -> TaskResult | None:
        """Full decompose -> route each subtask (depth 1) -> compose flow.

        Returns None when decomposition does not apply, so the caller can fall
        back to routing the task whole.
        """
        subtasks = self.split(prompt)
        if not subtasks:
            return None

        answers: list[tuple[str, str]] = []
        total_remote = 0
        traces = []
        worst_rung = Rung.CLASSIFY
        for subtask in subtasks:  # depth 1: subtasks are routed, never re-split
            sub_result = ladder.route(subtask)
            answers.append((subtask, sub_result.answer))
            total_remote += sub_result.remote_tokens
            traces.extend(sub_result.trace)
            worst_rung = max(worst_rung, sub_result.exit_rung)

        final = self.compose(prompt, answers)
        if not final:
            return None  # nothing usable; caller routes the task whole
        return TaskResult(
            answer=final,
            exit_rung=worst_rung,
            confidence=0.7,
            remote_tokens=total_remote,
            task_type=classify(prompt).task_type,
            verified=True,
            trace=tuple(traces),
        )


def _parse_subtasks(text: str) -> list[str]:
    """Extract a JSON array of strings from possibly messy model output."""
    match = _JSON_ARRAY.search(text)
    if not match:
        return []
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [item.strip() for item in parsed if isinstance(item, str) and item.strip()]
