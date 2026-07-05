"""Confidence, adaptive thresholds, and compression tests."""

import math

from routing_agent.router.adaptive import AdaptiveThresholds
from routing_agent.router.compression import compress_prompt
from routing_agent.router.confidence import logprob_to_confidence
from routing_agent.types import TaskType


def test_confidence_maps_logprob_exponentially():
    assert logprob_to_confidence(-0.1) == math.exp(-0.1)


def test_confidence_none_is_neutral():
    assert logprob_to_confidence(None) == 0.5


def test_confidence_clamps_positive_logprob():
    assert logprob_to_confidence(0.5) == 1.0


def test_adaptive_threshold_drops_after_successes():
    thresholds = AdaptiveThresholds(base_threshold=0.72)
    baseline = thresholds.get(TaskType.MATH)
    for _ in range(10):
        thresholds.update(TaskType.MATH, True)
    assert thresholds.get(TaskType.MATH) < baseline


def test_adaptive_threshold_rises_after_failures():
    thresholds = AdaptiveThresholds(base_threshold=0.72)
    baseline = thresholds.get(TaskType.CODE)
    for _ in range(10):
        thresholds.update(TaskType.CODE, False)
    assert thresholds.get(TaskType.CODE) > baseline


def test_adaptive_types_are_independent():
    thresholds = AdaptiveThresholds(base_threshold=0.72)
    for _ in range(10):
        thresholds.update(TaskType.MATH, True)
    assert thresholds.get(TaskType.CODE) == 0.72


def test_compress_strips_courtesy_and_duplicate_lines():
    prompt = "Please could you solve this.\nSame line\nSame line\n\n\n\nEnd?"
    compressed = compress_prompt(prompt)
    assert "Please" not in compressed and "could you" not in compressed
    assert compressed.count("Same line") == 1
    assert "\n\n\n" not in compressed


def test_compress_truncates_middle_when_over_budget():
    prompt = "INSTRUCTIONS " + "filler " * 5000 + " FINAL QUESTION?"
    compressed = compress_prompt(prompt, max_chars=1000)
    assert len(compressed) <= 1000
    assert compressed.startswith("INSTRUCTIONS")
    assert compressed.endswith("FINAL QUESTION?")
    assert "[...trimmed...]" in compressed


def test_compress_preserves_short_prompts():
    assert compress_prompt("What is 2+2?") == "What is 2+2?"
