"""Rung 0: zero-token heuristic task classification and difficulty scoring."""

from __future__ import annotations

import re

from routing_agent.types import Classification, TaskType

_MCQ_OPTION = re.compile(r"(?m)^\s*\(?([A-Ea-e])[).:]\s+\S")
_MATH_EXPR = re.compile(r"\d+\s*[-+*/^%=]\s*\d+")
_NUMBER = re.compile(r"\d")
_CODE_FENCE = re.compile(r"```")

_MATH_WORDS = (
    "calculate", "compute", "solve", "how many", "how much", "sum of", "product of",
    "percent", "average", "remainder", "divided by", "equation", "probability",
)
_CODE_WORDS = (
    "write a function", "write code", "implement", "python", "javascript",
    "regex", "sql query", "def ", "return a", "algorithm",
    # Debugging phrasings - the judged code-debugging category rarely says
    # "write code"; it says the code is broken.
    "debug", "fix the bug", "fix this code", "fix the code",
    "what's wrong with", "why does this fail", "throws an error",
    "doesn't work", "does not work", "corrected implementation",
)
_EXTRACTION_WORDS = (
    "extract", "list all", "find all", "identify the", "pull out", "from the text",
    "from the following", "named entities",
    # NER phrasings without the word "extract".
    "people mentioned", "organizations mentioned", "locations mentioned",
    "dates mentioned", "companies mentioned", "entities in",
    "who are the people", "name the people", "name the organizations",
)
# Unfenced code snippets: a line opening a function/class definition.
_BARE_CODE_LINE = re.compile(
    r"(?m)^\s*(def |class |function |public |private |const |let |var )\w*"
)
# A fence alone is context, not intent; these verbs make it a code task.
_CODE_INTENT_WORDS = (
    "write", "implement", "fix", "debug", "correct", "refactor", "complete",
    "modify", "wrong", "bug", "error", "fail",
)
_SUMMARY_WORDS = ("summarize", "summarise", "tl;dr", "in one sentence", "briefly describe")
_SENTIMENT_WORDS = (
    "sentiment", "positive or negative", "positive, negative", "tone of the",
    "classify the review", "classify this review", "emotion expressed",
    "opinion expressed", "how does the author feel",
)
_LOGIC_WORDS = (
    # Deliberately specific phrases: bare conditionals like "if all" appear in
    # ordinary math word problems and would misroute them to the LOGIC tier.
    "logic puzzle", "logically", "deduce", "deduction", "syllogism",
    "must be true", "cannot be true", "who is telling the truth",
    "knights and knaves", "sits next to", "seated in", "is taller than",
    "satisfy all", "constraints:",
)
_HARD_MARKERS = (
    "prove", "derive", "step by step", "explain why", "explain in detail",
    "compare and contrast", "trade-off", "essay", "comprehensive",
)


def classify(prompt: str) -> Classification:
    """Classify the task and estimate difficulty for the local model, for free."""
    lowered = prompt.lower()
    signals: list[str] = []

    task_type = _detect_type(prompt, lowered, signals)
    difficulty = _estimate_difficulty(prompt, lowered, task_type, signals)
    return Classification(
        task_type=task_type, difficulty=difficulty, signals=tuple(signals)
    )


def _detect_type(prompt: str, lowered: str, signals: list[str]) -> TaskType:
    if len(_MCQ_OPTION.findall(prompt)) >= 2:
        signals.append("mcq-options")
        return TaskType.MCQ
    if any(word in lowered for word in _SENTIMENT_WORDS):
        signals.append("sentiment-keyword")
        return TaskType.SENTIMENT
    if any(word in lowered for word in _SUMMARY_WORDS):
        signals.append("summary-keyword")
        return TaskType.SUMMARY
    has_snippet = bool(_CODE_FENCE.search(prompt) or _BARE_CODE_LINE.search(prompt))
    has_code_intent = any(word in lowered for word in _CODE_INTENT_WORDS)
    if any(word in lowered for word in _CODE_WORDS) or (has_snippet and has_code_intent):
        signals.append("code-keyword")
        return TaskType.CODE
    if any(word in lowered for word in _EXTRACTION_WORDS):
        signals.append("extraction-keyword")
        return TaskType.EXTRACTION
    if any(word in lowered for word in _LOGIC_WORDS):
        signals.append("logic-keyword")
        return TaskType.LOGIC
    if _MATH_EXPR.search(prompt) or any(word in lowered for word in _MATH_WORDS):
        signals.append("math-signal")
        return TaskType.MATH
    if "?" in prompt:
        signals.append("question-mark")
        return TaskType.QA
    return TaskType.GENERAL

# Baseline difficulty per type, calibrated for a 1-4B local model.
_TYPE_BASE: dict[TaskType, float] = {
    TaskType.SENTIMENT: 0.20,
    TaskType.MCQ: 0.25,
    TaskType.EXTRACTION: 0.30,
    TaskType.QA: 0.35,
    TaskType.SUMMARY: 0.40,
    TaskType.MATH: 0.45,
    TaskType.GENERAL: 0.45,
    TaskType.CODE: 0.55,
    TaskType.LOGIC: 0.65,
}


def _estimate_difficulty(
    prompt: str, lowered: str, task_type: TaskType, signals: list[str]
) -> float:
    score = _TYPE_BASE[task_type]

    words = len(prompt.split())
    if words > 400:
        score += 0.25
        signals.append("very-long-prompt")
    elif words > 150:
        score += 0.10
        signals.append("long-prompt")

    hard_hits = sum(1 for marker in _HARD_MARKERS if marker in lowered)
    if hard_hits:
        score += min(0.15 * hard_hits, 0.30)
        signals.append(f"hard-markers:{hard_hits}")

    if prompt.count("?") > 2:
        score += 0.10
        signals.append("multi-question")

    if task_type == TaskType.MATH and len(_NUMBER.findall(prompt)) > 12:
        score += 0.10
        signals.append("many-numbers")

    return max(0.0, min(1.0, score))
