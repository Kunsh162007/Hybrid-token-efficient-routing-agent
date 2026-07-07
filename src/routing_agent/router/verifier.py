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
_SENTIMENT_LABELS = ("positive", "negative", "neutral", "mixed")
_MIN_SENTIMENT_WORDS = 4  # label alone is not the label-plus-justification the judge wants
_SENTENCE_LIMIT = re.compile(
    r"\bin (?:exactly )?(one|a single|two|three|1|2|3) sentences?\b", re.IGNORECASE
)
_WORD_LIMIT = re.compile(
    r"\b(?:in|under|at most|no more than|fewer than|less than|within|maximum(?: of)?)"
    r"\s+(\d{1,3})\s+words\b",
    re.IGNORECASE,
)
_SENTENCE_WORDS = {"one": 1, "a single": 1, "1": 1, "two": 2, "2": 2, "three": 3, "3": 3}
_SENTENCE_END = re.compile(r"[.!?]+(?:\s|$)")


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
    if task_type == TaskType.SENTIMENT:
        # Vote and score on the label; the justification wording may vary.
        # Earliest mention wins: "Negative, not positive at all" is negative.
        lowered = final.lower()
        hits = [
            (lowered.find(label), label)
            for label in _SENTIMENT_LABELS
            if label in lowered
        ]
        if hits:
            return min(hits)[1]
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
        constraint = _summary_constraint_violation(prompt, stripped)
        if constraint:
            return VerifyResult(ok=False, normalized=normalized, reason=constraint)
    elif task_type == TaskType.SENTIMENT:
        if normalized not in _SENTIMENT_LABELS:
            return VerifyResult(
                ok=False, normalized=normalized, reason="no sentiment label"
            )
        if len(stripped.split()) < _MIN_SENTIMENT_WORDS:
            return VerifyResult(
                ok=False, normalized=normalized, reason="missing justification"
            )

    if len(stripped) > 8000:
        return VerifyResult(ok=False, normalized=normalized, reason="answer absurdly long")

    return VerifyResult(ok=True, normalized=normalized)


def _summary_constraint_violation(prompt: str, answer: str) -> str:
    """Check the answer against an explicit length constraint in the prompt.

    Returns a failure reason, or "" when no constraint is stated or it holds.
    The judged summarisation category scores format/length compliance, so a
    5-sentence answer to an 'in one sentence' prompt must not ship.
    """
    sentence_match = _SENTENCE_LIMIT.search(prompt)
    if sentence_match:
        limit = _SENTENCE_WORDS[sentence_match.group(1).lower()]
        count = max(len(_SENTENCE_END.findall(answer)), 1)
        if count > limit:
            return f"{count} sentences where the task asked for {limit}"
    word_match = _WORD_LIMIT.search(prompt)
    if word_match:
        limit = int(word_match.group(1))
        count = len(answer.split())
        if count > limit:
            return f"{count} words where the task allowed {limit}"
    return ""


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
