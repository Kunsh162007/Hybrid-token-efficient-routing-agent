"""Learned rung-0 router: logistic regression over cheap features.

Trains on eval-harness records (label = 'local ladder solved it') and predicts
P(local succeeds) for new prompts. It replaces the heuristic difficulty score
only when enabled in config; heuristics always remain the fallback.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from routing_agent.router.classifier import classify
from routing_agent.types import Classification, TaskType

_TYPES = list(TaskType)


class LearnedRouterUnavailable(Exception):
    """scikit-learn missing, model file absent, or not enough training data."""


def _features(prompt: str) -> np.ndarray:
    """Cheap, deterministic features: no tokens, no network."""
    cls = classify(prompt)
    words = min(len(prompt.split()) / 200.0, 2.0)
    questions = min(prompt.count("?") / 3.0, 1.0)
    type_onehot = [1.0 if cls.task_type == t else 0.0 for t in _TYPES]
    return np.asarray(
        [words, cls.difficulty, questions, *type_onehot], dtype=np.float64
    )


class LearnedRouter:
    """Wraps a fitted LogisticRegression predicting P(local success)."""

    def __init__(self, model) -> None:
        self._model = model

    @classmethod
    def train(cls, records: list[dict], *, min_records: int = 20) -> "LearnedRouter":
        """Fit from harness records: [{'prompt': ..., 'label': 0|1}, ...]."""
        try:
            from sklearn.linear_model import LogisticRegression
        except ImportError as exc:
            raise LearnedRouterUnavailable(
                "scikit-learn not installed; install with pip install '.[learned]'"
            ) from exc
        usable = [r for r in records if r.get("prompt") and r.get("label") is not None]
        if len(usable) < min_records:
            raise LearnedRouterUnavailable(
                f"Need at least {min_records} training records, have {len(usable)}"
            )
        labels = np.asarray([int(r["label"]) for r in usable])
        if len(set(labels.tolist())) < 2:
            raise LearnedRouterUnavailable("Training data has only one class")
        matrix = np.stack([_features(r["prompt"]) for r in usable])
        model = LogisticRegression(max_iter=1000)
        model.fit(matrix, labels)
        return cls(model)

    @classmethod
    def train_from_log(cls, log_path: str | Path) -> "LearnedRouter":
        path = Path(log_path)
        if not path.exists():
            raise LearnedRouterUnavailable(f"No training log at {path}")
        records = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return cls.train(records)

    def predict_p_local(self, prompt: str) -> float:
        proba = self._model.predict_proba(_features(prompt).reshape(1, -1))[0]
        positive_index = list(self._model.classes_).index(1)
        return float(proba[positive_index])

    def as_estimator(self):
        """Adapter: plug into EscalationLadder(difficulty_estimator=...)."""

        def estimate(prompt: str) -> Classification:
            heuristic = classify(prompt)
            p_local = self.predict_p_local(prompt)
            return Classification(
                task_type=heuristic.task_type,
                difficulty=1.0 - p_local,
                signals=(*heuristic.signals, f"learned:p_local={p_local:.2f}"),
            )

        return estimate

    def save(self, path: str | Path) -> None:
        import joblib

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._model, target)

    @classmethod
    def load(cls, path: str | Path) -> "LearnedRouter":
        try:
            import joblib
        except ImportError as exc:
            raise LearnedRouterUnavailable("joblib not installed") from exc
        target = Path(path)
        if not target.exists():
            raise LearnedRouterUnavailable(f"No trained router at {target}")
        return cls(joblib.load(target))
