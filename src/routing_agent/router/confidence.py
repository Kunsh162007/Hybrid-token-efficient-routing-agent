"""Confidence scoring from local-model logprobs.

exp(mean token logprob) is the geometric-mean per-token probability: an
honest, free confidence signal, unlike asking the model 'are you sure?'.
"""

from __future__ import annotations

import math

NEUTRAL_CONFIDENCE = 0.5


def logprob_to_confidence(logprob_mean: float | None) -> float:
    """Map mean token logprob to [0, 1]; None (no logprobs) is neutral."""
    if logprob_mean is None:
        return NEUTRAL_CONFIDENCE
    if logprob_mean > 0:  # defensive: logprobs should never be positive
        return 1.0
    return math.exp(logprob_mean)
