"""Local text embeddings for the semantic cache.

Uses fastembed (ONNX, CPU, no torch) when available. If it is not installed,
build_embedder returns None and the cache silently degrades to exact-only,
which is always correct, just less generous.
"""

from __future__ import annotations

import numpy as np


class Embedder:
    """Thin wrapper so the cache depends on one tiny surface."""

    def __init__(self, model_name: str) -> None:
        from fastembed import TextEmbedding  # deferred heavy import

        self._model = TextEmbedding(model_name=model_name)

    def embed(self, text: str) -> np.ndarray:
        vector = next(iter(self._model.embed([text])))
        array = np.asarray(vector, dtype=np.float32)
        norm = np.linalg.norm(array)
        return array / norm if norm > 0 else array


def build_embedder(model_name: str) -> Embedder | None:
    """Return an Embedder, or None when fastembed is unavailable."""
    try:
        return Embedder(model_name)
    except ImportError:
        return None
    except Exception:
        # Model download failure, unsupported platform, etc. Cache degrades.
        return None
