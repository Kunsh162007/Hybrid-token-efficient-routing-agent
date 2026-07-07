"""Deterministic zero-token solving for explicit arithmetic.

A language model is the wrong tool for '128 * 46'. When a MATH task contains
an explicit expression or a 'sum/average/product of ...' phrasing, Python
computes the exact answer for free - no tokens, no sampling, no doubt.
Anything ambiguous returns None and takes the normal ladder.
"""

from __future__ import annotations

import ast
import math
import operator
import re

_AGGREGATE = re.compile(
    r"(average|mean|sum|product) of ((?:\d[\d,\.]*(?:\s*,\s*|\s+and\s+|\s+)?)+)",
    re.IGNORECASE,
)
_CANDIDATE = re.compile(r"[\d(][\d\s.,+\-*/x×^%()]*\d")
# Text continuing an aggregate with more arithmetic ("..., minus 2") means the
# captured span is only part of the expression - bail rather than be wrong.
_AGG_CONTINUATION = re.compile(
    r"^\s*,?\s*(minus|plus|times|multiplied|divided|[-+*/×^%])", re.IGNORECASE
)
_STRONG_OP = re.compile(r"[*/×^%]|x(?=\s*\d)")
_PLUS_MINUS = re.compile(r"[+\-]")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.USub: operator.neg, ast.UAdd: operator.pos}

_MAX_POW_EXPONENT = 64  # keep pow from generating astronomically long output


def try_solve_math(prompt: str) -> str | None:
    """Exact answer for explicit arithmetic, or None when unsure."""
    aggregate = _solve_aggregate(prompt)
    if aggregate is not None:
        return aggregate

    best: tuple[int, float] | None = None
    for match in _CANDIDATE.finditer(prompt):
        text = match.group(0).strip()
        if not _is_confident_expression(text):
            continue
        value = _evaluate(text)
        if value is None:
            continue
        if best is None or len(text) > best[0]:
            best = (len(text), value)
    return _format(best[1]) if best else None


def _solve_aggregate(prompt: str) -> str | None:
    match = _AGGREGATE.search(prompt)
    if not match:
        return None
    if _AGG_CONTINUATION.match(prompt[match.end():]):
        return None
    numbers = [float(n) for n in _NUMBER.findall(match.group(2))]
    if len(numbers) < 2:
        return None
    kind = match.group(1).lower()
    if kind in ("average", "mean"):
        return _format(sum(numbers) / len(numbers))
    if kind == "sum":
        return _format(sum(numbers))
    return _format(math.prod(numbers))


def _is_confident_expression(text: str) -> bool:
    """Fire only on unambiguous math: strong operators, or 2+ plus/minus."""
    if not _NUMBER.search(text):
        return False
    if _STRONG_OP.search(text):
        return True
    return len(_PLUS_MINUS.findall(text)) >= 2


def _evaluate(text: str) -> float | None:
    cleaned = (
        text.replace(",", "").replace("×", "*").replace("^", "**")
    )
    cleaned = re.sub(r"x(?=\s*\d)", "*", cleaned)
    try:
        tree = ast.parse(cleaned.strip(), mode="eval")
        return _safe_eval(tree.body)
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError):
        return None


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left, right = _safe_eval(node.left), _safe_eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > _MAX_POW_EXPONENT:
            raise ValueError("exponent too large")
        return _BIN_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"unsupported node: {type(node).__name__}")


def _format(value: float) -> str:
    if math.isfinite(value) and float(value).is_integer():
        return str(int(value))
    return str(round(value, 6))
