"""Prompt compression for remote escalation.

Every character stripped here is an input token never billed. Compression is
conservative: it removes redundancy, never task content.
"""

from __future__ import annotations

import re

_COURTESY = re.compile(
    r"(?i)\b(please|kindly|could you|would you|i would like you to|"
    r"can you help me|if you don'?t mind)\b\s*",
)
_MULTI_SPACE = re.compile(r"[ \t]{2,}")
_MULTI_NEWLINE = re.compile(r"\n{3,}")

TRUNCATION_MARKER = "\n[...trimmed...]\n"


def compress_prompt(prompt: str, max_chars: int = 6000) -> str:
    """Strip redundancy; middle-truncate only when far over budget."""
    text = _COURTESY.sub("", prompt)
    text = _MULTI_SPACE.sub(" ", text)
    text = _dedupe_adjacent_lines(text)
    text = _MULTI_NEWLINE.sub("\n\n", text).strip()

    if len(text) > max_chars:
        # Keep head and tail: instructions usually live at the edges.
        head = int(max_chars * 0.65)
        tail = max_chars - head - len(TRUNCATION_MARKER)
        text = text[:head] + TRUNCATION_MARKER + text[-tail:]
    return text


def _dedupe_adjacent_lines(text: str) -> str:
    lines = text.split("\n")
    kept: list[str] = []
    for line in lines:
        if kept and line.strip() and line.strip() == kept[-1].strip():
            continue
        kept.append(line)
    return "\n".join(kept)
