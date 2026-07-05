"""Answer cache tests with a deterministic fake embedder."""

import numpy as np

from routing_agent.cache.store import AnswerCache
from routing_agent.config import CacheConfig


class FakeEmbedder:
    """Maps known phrases to fixed unit vectors so similarity is controllable."""

    def __init__(self, table):
        self.table = table

    def embed(self, text):
        for phrase, vector in self.table.items():
            if phrase in text:
                array = np.asarray(vector, dtype=np.float32)
                return array / np.linalg.norm(array)
        return np.asarray([0.0, 0.0, 1.0], dtype=np.float32)


def make_cache(tmp_path, embedder=None, threshold=0.95):
    config = CacheConfig(
        db_path=str(tmp_path / "cache.db"), semantic_threshold=threshold
    )
    return AnswerCache(config, embedder=embedder if embedder is not None else _NoEmbedder())


class _NoEmbedder:
    """Sentinel meaning 'exact-only': AnswerCache treats non-None as an embedder,
    so tests that want exact-only pass an embedder that never matches."""

    def embed(self, text):
        return np.asarray([1.0, 0.0, 0.0], dtype=np.float32)


def test_exact_hit_after_put(tmp_path):
    cache = make_cache(tmp_path)
    cache.put("What is 2+2?", "4")
    assert cache.lookup("What is 2+2?") == "4"


def test_exact_hit_is_case_and_whitespace_insensitive(tmp_path):
    cache = make_cache(tmp_path)
    cache.put("What is 2+2?", "4")
    assert cache.lookup("  what IS   2+2? ") == "4"


def test_miss_returns_none(tmp_path):
    cache = make_cache(tmp_path)
    assert cache.lookup("never seen") is None


def test_semantic_hit_above_threshold(tmp_path):
    embedder = FakeEmbedder({
        "capital of france": [1.0, 0.0, 0.0],
        "france's capital": [0.99, 0.14, 0.0],  # cosine ~0.99 with the above
    })
    cache = make_cache(tmp_path, embedder=embedder, threshold=0.95)
    cache.put("What is the capital of france?", "Paris")

    assert cache.lookup("Tell me france's capital") == "Paris"
    assert cache.semantic_hits == 1


def test_semantic_miss_below_threshold(tmp_path):
    embedder = FakeEmbedder({
        "capital of france": [1.0, 0.0, 0.0],
        "boiling point": [0.0, 1.0, 0.0],  # orthogonal
    })
    cache = make_cache(tmp_path, embedder=embedder)
    cache.put("What is the capital of france?", "Paris")

    assert cache.lookup("What is the boiling point of water?") is None


def test_cache_persists_across_instances(tmp_path):
    config = CacheConfig(db_path=str(tmp_path / "cache.db"))
    first = AnswerCache(config, embedder=_NoEmbedder())
    first.put("q1", "a1")
    first.close()

    second = AnswerCache(config, embedder=_NoEmbedder())
    assert second.lookup("q1") == "a1"
    assert second.size() == 1


def test_put_overwrites_same_prompt(tmp_path):
    cache = make_cache(tmp_path)
    cache.put("q", "old")
    cache.put("q", "new")
    assert cache.lookup("q") == "new"
    assert cache.size() == 1
