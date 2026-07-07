"""Heuristic classifier tests."""

from routing_agent.router.classifier import classify
from routing_agent.types import TaskType


def test_detects_mcq_from_lettered_options():
    prompt = "Which is a prime?\nA) 4\nB) 7\nC) 9\nD) 12"
    cls = classify(prompt)
    assert cls.task_type == TaskType.MCQ


def test_detects_math_from_expression():
    assert classify("What is 128 * 46?").task_type == TaskType.MATH


def test_detects_math_from_keywords():
    assert classify("Calculate the average of the values.").task_type == TaskType.MATH


def test_detects_code():
    assert classify("Write a function to reverse a string in Python.").task_type == TaskType.CODE


def test_detects_extraction():
    prompt = "Extract all dates from the following text: ..."
    assert classify(prompt).task_type == TaskType.EXTRACTION


def test_detects_code_debugging_phrasing():
    prompt = "What's wrong with this code? It throws an error on empty input."
    assert classify(prompt).task_type == TaskType.CODE


def test_detects_ner_without_extract_keyword():
    prompt = "Who are the people and organizations mentioned in this article?"
    assert classify(prompt).task_type == TaskType.EXTRACTION


def test_code_fence_without_intent_is_not_code():
    prompt = "What does this snippet output? ```\nconsole.log(1 + 1)\n```"
    assert classify(prompt).task_type != TaskType.CODE


def test_detects_summary():
    assert classify("Summarize this article: ...").task_type == TaskType.SUMMARY


def test_detects_sentiment():
    prompt = "Classify the sentiment of this review and justify: 'Great phone!'"
    assert classify(prompt).task_type == TaskType.SENTIMENT


def test_detects_logic_puzzle():
    prompt = (
        "Alice, Bob and Carol are seated in a row. Alice sits next to Bob "
        "but not Carol. Deduce who sits in the middle."
    )
    assert classify(prompt).task_type == TaskType.LOGIC


def test_math_word_problem_with_conditional_stays_math():
    prompt = "If all the boxes weigh 5 kg each, how much do 3 boxes weigh?"
    assert classify(prompt).task_type == TaskType.MATH


def test_logic_is_harder_than_sentiment():
    logic = classify("Deduce who is telling the truth in this logic puzzle.")
    sentiment = classify("What is the sentiment of: 'I love it'?")
    assert logic.difficulty > sentiment.difficulty


def test_question_falls_back_to_qa():
    assert classify("What is the capital of France?").task_type == TaskType.QA


def test_plain_instruction_is_general():
    assert classify("Translate hello to French.").task_type == TaskType.GENERAL


def test_long_prompts_are_harder():
    short = classify("What is 2+2?")
    long = classify("What is 2+2? " + "context " * 500)
    assert long.difficulty > short.difficulty


def test_hard_markers_raise_difficulty():
    easy = classify("What is the capital of France?")
    hard = classify("Explain why the capital of France changed over time, step by step?")
    assert hard.difficulty > easy.difficulty


def test_difficulty_stays_in_unit_range():
    monster = "Prove and derive step by step, explain why. " * 200 + "?"
    assert 0.0 <= classify(monster).difficulty <= 1.0
