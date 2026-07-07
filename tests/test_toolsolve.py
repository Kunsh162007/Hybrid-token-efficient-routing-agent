"""Deterministic math tool tests."""

from routing_agent.router.toolsolve import try_solve_math


def test_solves_explicit_multiplication():
    assert try_solve_math("What is 128 * 46?") == "5888"


def test_solves_with_commas_and_unicode_times():
    assert try_solve_math("Compute 1,200 × 3") == "3600"


def test_solves_caret_power():
    assert try_solve_math("What is 2^10?") == "1024"


def test_solves_average_phrase():
    assert try_solve_math("Calculate the average of 10, 20, and 30.") == "20"


def test_solves_sum_phrase():
    assert try_solve_math("What is the sum of 5, 10 and 15?") == "30"


def test_solves_product_phrase():
    assert try_solve_math("Find the product of 3, 4 and 5.") == "60"


def test_aggregate_with_trailing_arithmetic_bails_out():
    # "sum of 12, 5 and 3, minus 2" is 18; committing to 20 would be a silent
    # wrong answer at 0.99 confidence, so the solver must decline.
    assert try_solve_math("What is the sum of 12, 5 and 3, minus 2?") is None


def test_division_formats_decimals():
    assert try_solve_math("What is 7 / 2?") == "3.5"


def test_word_problem_returns_none():
    # Numbers exist but no explicit expression: the model must reason.
    assert try_solve_math(
        "A shop sells pens at 12 rupees each. How much do 7 pens cost?"
    ) is None


def test_single_plus_minus_is_not_confident():
    # '10 - 20' could be a range; needs stronger evidence.
    assert try_solve_math("Prices go from 10 - 20 rupees.") is None


def test_multiple_plus_operators_are_confident():
    assert try_solve_math("What is 1 + 2 + 3?") == "6"


def test_huge_exponent_refused():
    assert try_solve_math("What is 9 ^ 999999?") is None


def test_division_by_zero_refused():
    assert try_solve_math("What is 5 / 0?") is None


def test_no_numbers_returns_none():
    assert try_solve_math("What is the capital of France?") is None
