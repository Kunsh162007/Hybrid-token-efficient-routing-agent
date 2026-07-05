"""Zero-token answer verification and normalization.

The verifier is the gatekeeper between free local answers and paid escalation:
a strict-but-fair check that costs nothing.
"""

from __future__ import annotations

import re
from collections import Counter

from routing_agent.types import TaskType, VerifyResult

_ANSWER_MARKER = re.compile(r"(?is)\banswer\s*[:=]\s*")
_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_MCQ_LETTER = re.compile(r"\b([A-Ea-e])\b")
_CODE_BLOCK = re.compile(r"```(?:\w+)?\s*\n?(.*?)```", re.DOTALL)
_REFUSAL_MARKERS = (
    "i cannot", "i can't", "i'm unable", "as an ai", "i am unable",
    "i don't have access", "i'm sorry, but",
)


def extract_final(text: str) -> str:
    """Take the part after the last 'Answer:' marker, if the model used one."""
    parts = _ANSWER_MARKER.split(text)
    return parts[-1].strip() if len(parts) > 1 else text.strip()


def normalize(task_type: TaskType, text: str) -> str:
    """Canonical comparable form of an answer, for voting and scoring."""
    final = extract_final(text)
    if task_type == TaskType.MATH:
        numbers = _NUMBER.findall(final) or _NUMBER.findall(text)
        return numbers[-1].replace(",", "") if numbers else final.lower().strip()
    if task_type == TaskType.MCQ:
        match = _MCQ_LETTER.search(final) or _MCQ_LETTER.search(text)
        return match.group(1).upper() if match else final.upper().strip()
    if task_type == TaskType.CODE:
        blocks = _CODE_BLOCK.findall(text)
        code = blocks[0] if blocks else final
        return code.strip()
    return " ".join(final.lower().split())


def verify(task_type: TaskType, prompt: str, answer: str) -> VerifyResult:
    """Cheap validity check; failing it forces the ladder to climb."""
    stripped = answer.strip()
    if not stripped:
        return VerifyResult(ok=False, normalized="", reason="empty answer")

    lowered = stripped.lower()
    if any(marker in lowered[:120] for marker in _REFUSAL_MARKERS):
        return VerifyResult(ok=False, normalized="", reason="refusal")

    normalized = normalize(task_type, answer)

    if task_type == TaskType.MATH:
        if not _NUMBER.fullmatch(normalized):
            return VerifyResult(ok=False, normalized=normalized, reason="no numeric answer")
    elif task_type == TaskType.MCQ:
        if normalized not in {"A", "B", "C", "D", "E"}:
            return VerifyResult(ok=False, normalized=normalized, reason="no option letter")
        offered = {
            letter.upper()
            for letter in re.findall(r"(?m)^\s*\(?([A-Ea-e])[).:]\s+", prompt)
        }
        if offered and normalized not in offered:
            return VerifyResult(
                ok=False, normalized=normalized, reason="letter not among options"
            )
    elif task_type == TaskType.CODE:
        if not _looks_like_valid_code(normalized, prompt):
            return VerifyResult(ok=False, normalized=normalized, reason="code does not parse")
    elif task_type == TaskType.SUMMARY:
        if len(normalized.split()) > max(len(prompt.split()), 30):
            return VerifyResult(
                ok=False, normalized=normalized, reason="summary longer than source"
            )

    if len(stripped) > 8000:
        return VerifyResult(ok=False, normalized=normalized, reason="answer absurdly long")

    return VerifyResult(ok=True, normalized=normalized)


def majority_vote(task_type: TaskType, answers: list[str]) -> tuple[str, float]:
    """Self-consistency vote over normalized answers.

    Returns (winning raw answer, vote ratio). Free consensus signal.
    """
    if not answers:
        return "", 0.0
    normalized = [normalize(task_type, answer) for answer in answers]
    counts = Counter(norm for norm in normalized if norm)
    if not counts:
        return answers[0], 1.0 / len(answers)
    winner_norm, winner_count = counts.most_common(1)[0]
    winner_raw = next(
        raw for raw, norm in zip(answers, normalized, strict=True) if norm == winner_norm
    )
    return winner_raw, winner_count / len(answers)


def _looks_like_valid_code(code: str, prompt: str) -> bool:
    """Python answers must compile; other languages get a shape check."""
    if not code:
        return False
    prompt_lower = prompt.lower()
    wants_python = "python" in prompt_lower or "def " in code
    if wants_python:
        try:
            compile(code, "<candidate>", "exec")
            return True
        except SyntaxError:
            return False
    return len(code.splitlines()) >= 1
