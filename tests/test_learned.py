"""Learned router tests (skipped when scikit-learn is unavailable)."""

import pytest

pytest.importorskip("sklearn")

from routing_agent.router.learned import LearnedRouter, LearnedRouterUnavailable
from routing_agent.types import TaskType

# Synthetic training data with a clean signal: short math succeeds locally,
# long "explain" prompts do not.
EASY = [{"prompt": f"What is {i}+{i}?", "label": 1} for i in range(2, 22)]
HARD = [
    {"prompt": "Explain in detail step by step why " + "context " * 200 + f"case {i}?",
     "label": 0}
    for i in range(20)
]


def test_train_and_predict_separates_easy_from_hard():
    router = LearnedRouter.train(EASY + HARD)

    p_easy = router.predict_p_local("What is 9+9?")
    p_hard = router.predict_p_local(
        "Explain in detail step by step why " + "context " * 200 + "this happens?"
    )
    assert p_easy > p_hard


def test_as_estimator_returns_classification_with_learned_difficulty():
    router = LearnedRouter.train(EASY + HARD)
    estimate = router.as_estimator()

    cls = estimate("What is 5+5?")
    assert cls.task_type == TaskType.MATH
    assert 0.0 <= cls.difficulty <= 1.0
    assert any(s.startswith("learned:") for s in cls.signals)


def test_train_requires_minimum_records():
    with pytest.raises(LearnedRouterUnavailable, match="at least"):
        LearnedRouter.train(EASY[:5])


def test_train_requires_both_classes():
    with pytest.raises(LearnedRouterUnavailable, match="one class"):
        LearnedRouter.train(EASY + EASY)


def test_save_load_roundtrip(tmp_path):
    router = LearnedRouter.train(EASY + HARD)
    path = tmp_path / "model" / "router.joblib"
    router.save(path)

    loaded = LearnedRouter.load(path)
    prompt = "What is 7+7?"
    assert abs(loaded.predict_p_local(prompt) - router.predict_p_local(prompt)) < 1e-9


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(LearnedRouterUnavailable, match="No trained router"):
        LearnedRouter.load(tmp_path / "absent.joblib")
