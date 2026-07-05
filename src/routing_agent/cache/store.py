"""SQLite-backed answer cache with exact and semantic lookup.

Only *paid* remote answers are stored: local answers are free to regenerate.
An exact hash hit costs nothing; a semantic hit reuses a paid answer for a
near-duplicate query. Every hit still passes the ladder's verifier before use.
"""

from __future__ import annotations

import hashlib
import sqlite3
import threading
import time
from pathlib import Path

import numpy as np

from routing_agent.cache.embeddings import build_embedder
from routing_agent.config import CacheConfig

_SCHEMA = """
CREATE TABLE IF NOT EXISTS answers (
    prompt_hash TEXT PRIMARY KEY,
    prompt      TEXT NOT NULL,
    answer      TEXT NOT NULL,
    embedding   BLOB,
    created_at  REAL NOT NULL
)
"""


class AnswerCache:
    """Exact-first, semantic-second cache. Thread-safe."""

    def __init__(self, config: CacheConfig, *, embedder=None) -> None:
        self._config = config
        Path(config.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(config.db_path, check_same_thread=False)
        self._conn.execute(_SCHEMA)
        self._conn.commit()
        self._lock = threading.Lock()
        self._embedder = embedder if embedder is not None else build_embedder(
            config.embedding_model
        )
        self.hits = 0
        self.semantic_hits = 0

    @property
    def semantic_enabled(self) -> bool:
        return self._embedder is not None

    def lookup(self, prompt: str) -> str | None:
        """Exact hash first, then cosine similarity over stored embeddings."""
        key = _hash(prompt)
        with self._lock:
            row = self._conn.execute(
                "SELECT answer FROM answers WHERE prompt_hash = ?", (key,)
            ).fetchone()
        if row is not None:
            self.hits += 1
            return row[0]

        if self._embedder is None:
            return None
        query = self._embedder.embed(_canonical(prompt))
        with self._lock:
            rows = self._conn.execute(
                "SELECT answer, embedding FROM answers WHERE embedding IS NOT NULL"
            ).fetchall()
        best_answer, best_score = None, 0.0
        for answer, blob in rows:
            stored = np.frombuffer(blob, dtype=np.float32)
            if stored.shape != query.shape:
                continue
            score = float(np.dot(query, stored))  # both unit-normalized
            if score > best_score:
                best_answer, best_score = answer, score
        if best_answer is not None and best_score >= self._config.semantic_threshold:
            self.hits += 1
            self.semantic_hits += 1
            return best_answer
        return None

    def put(self, prompt: str, answer: str) -> None:
        embedding_blob = None
        if self._embedder is not None:
            embedding_blob = self._embedder.embed(_canonical(prompt)).tobytes()
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO answers VALUES (?, ?, ?, ?, ?)",
                (_hash(prompt), prompt, answer, embedding_blob, time.time()),
            )
            self._conn.commit()

    def size(self) -> int:
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM answers").fetchone()[0]

    def close(self) -> None:
        self._conn.close()


def _canonical(prompt: str) -> str:
    return " ".join(prompt.lower().split())


def _hash(prompt: str) -> str:
    return hashlib.sha256(_canonical(prompt).encode("utf-8")).hexdigest()
