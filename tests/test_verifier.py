"""Verifier and voting tests."""

from routing_agent.router.verifier import extract_final, majority_vote, normalize, verify
from routing_agent.types import TaskType


def test_extract_final_takes_text_after_answer_marker():
    assert extract_final("Let me think... Answer: 42") == "42"


def test_extract_final_without_marker_returns_all():
    assert extract_final("  just text  ") == "just text"


def test_normalize_math_takes_last_number():
    assert normalize(TaskType.MATH, "First 3, then 6, so Answer: 1,234") == "1234"


def test_normalize_mcq_takes_letter():
    assert normalize(TaskType.MCQ, "The best option is Answer: (b)") == "B"


def test_normalize_code_strips_fence():
    text = "```python\ndef f():\n    return 1\n```"
    assert normalize(TaskType.CODE, text).startswith("def f():")


def test_normalize_sentiment_maps_to_label():
    text = "Positive. The reviewer praises the battery life."
    assert normalize(TaskType.SENTIMENT, text) == "positive"


def test_normalize_sentiment_contrastive_phrasing_uses_stated_label():
    text = "Negative, not positive at all - the reviewer is furious."
    assert normalize(TaskType.SENTIMENT, text) == "negative"


def test_sentiment_vote_agrees_across_different_justifications():
    answers = [
        "Positive - the tone is enthusiastic.",
        "positive, because the reviewer recommends it",
        "The sentiment is positive; strong praise throughout.",
    ]
    winner, ratio = majority_vote(TaskType.SENTIMENT, answers)
    assert ratio == 1.0
    assert "positive" in winner.lower()


def test_verify_sentiment_requires_label():
    assert verify(TaskType.SENTIMENT, "Sentiment?", "Negative - harsh review.").ok
    assert not verify(TaskType.SENTIMENT, "Sentiment?", "It is a nice review.").ok


def test_verify_sentiment_bare_label_needs_justification():
    result = verify(TaskType.SENTIMENT, "Sentiment?", "Positive")
    assert not result.ok
    assert "justification" in result.reason


def test_verify_summary_enforces_sentence_constraint():
    prompt = "Summarise the following in one sentence: ..."
    long_answer = "First sentence. Second sentence. Third sentence."
    short_answer = "A single tidy sentence."
    assert not verify(TaskType.SUMMARY, prompt, long_answer).ok
    assert verify(TaskType.SUMMARY, prompt, short_answer).ok


def test_verify_summary_enforces_word_constraint():
    prompt = "Summarise this in under 5 words: ..."
    assert not verify(TaskType.SUMMARY, prompt, "This answer clearly has too many words.").ok
    assert verify(TaskType.SUMMARY, prompt, "Concise summary here.").ok


def test_paraphrased_summaries_vote_together():
    # Two good summaries are never word-identical; word-overlap must unify
    # them or every free-text vote splits and escalates to a paid rung.
    answers = [
        "The council approved a pilot program for bicycle lanes on Main Street.",
        "The council approved the bicycle lane pilot program on Main Street.",
    ]
    winner, ratio = majority_vote(TaskType.SUMMARY, answers)
    assert ratio == 1.0
    assert "council" in winner.lower()


def test_unrelated_free_text_answers_still_split_the_vote():
    answers = [
        "The council approved a bicycle lane pilot program.",
        "Parking fees will rise by two percent downtown next year.",
    ]
    _, ratio = majority_vote(TaskType.SUMMARY, answers)
    assert ratio == 0.5


def test_verify_rejects_empty():
    assert not verify(TaskType.QA, "q?", "   ").ok


def test_verify_rejects_refusal():
    assert not verify(TaskType.QA, "q?", "I cannot answer that question.").ok


def test_verify_math_requires_number():
    assert not verify(TaskType.MATH, "2+2?", "the answer is four-ish maybe").ok
    assert verify(TaskType.MATH, "2+2?", "Answer: 4").ok


def test_verify_mcq_requires_offered_letter():
    prompt = "Pick one:\nA) x\nB) y"
    assert verify(TaskType.MCQ, prompt, "Answer: B").ok
    assert not verify(TaskType.MCQ, prompt, "Answer: D").ok


def test_verify_python_code_must_compile():
    prompt = "Write a python function"
    good = "```python\ndef f():\n    return 1\n```"
    bad = "```python\ndef f(:\n```"
    assert verify(TaskType.CODE, prompt, good).ok
    assert not verify(TaskType.CODE, prompt, bad).ok


def test_verify_summary_must_be_shorter_than_source():
    prompt = "Summarize: " + "word " * 200
    assert verify(TaskType.SUMMARY, prompt, "Short summary.").ok
    assert not verify(TaskType.SUMMARY, prompt, "blah " * 300).ok


def test_majority_vote_picks_most_common_normalized():
    answers = ["Answer: 4", "Answer: 5", "the answer is 4", "Answer: 4"]
    winner, ratio = majority_vote(TaskType.MATH, answers)
    assert normalize(TaskType.MATH, winner) == "4"
    assert ratio == 0.75


def test_majority_vote_empty():
    winner, ratio = majority_vote(TaskType.MATH, [])
    assert winner == "" and ratio == 0.0
