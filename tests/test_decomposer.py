"""Decomposer guardrail and flow tests."""

from conftest import FakeLocalClient, FakeRemoteClient

from routing_agent.budget import BudgetTracker
from routing_agent.config import DecomposerConfig, LadderConfig, LocalModelConfig, RemoteModelConfig
from routing_agent.router.decomposer import Decomposer, _parse_subtasks
from routing_agent.router.ladder import EscalationLadder

LONG_TASK = (
    "First, compute the total cost of 3 pens at 12 rupees each. "
    "Second, compute the total cost of 2 books at 50 rupees each. "
    "Then report both totals so we can compare the two amounts clearly."
)


def make_ladder(local, remote):
    return EscalationLadder(
        LadderConfig(), LocalModelConfig(), RemoteModelConfig(),
        local, remote, BudgetTracker(per_task_budget=2000),
    )


def test_parse_subtasks_handles_messy_output():
    text = 'Sure! Here you go:\n["find pens cost", "find books cost"]\nDone.'
    assert _parse_subtasks(text) == ["find pens cost", "find books cost"]


def test_parse_subtasks_rejects_non_json():
    assert _parse_subtasks("no array here") == []
    assert _parse_subtasks("[1, 2, 3]") == []  # non-string items dropped


def test_disabled_decomposer_never_splits():
    local = FakeLocalClient(answers=['["a", "b"]'])
    decomposer = Decomposer(DecomposerConfig(enabled=False), local)
    assert decomposer.split(LONG_TASK) == []
    assert local.calls == []


def test_short_prompts_are_never_split():
    local = FakeLocalClient(answers=['["a", "b"]'])
    decomposer = Decomposer(DecomposerConfig(enabled=True), local)
    assert decomposer.split("What is 2+2?") == []
    assert local.calls == []


def test_split_caps_subtask_count():
    local = FakeLocalClient(answers=['["a", "b", "c", "d", "e", "f"]'])
    decomposer = Decomposer(DecomposerConfig(enabled=True, max_subtasks=4), local)
    assert len(decomposer.split(LONG_TASK)) == 4


def test_single_item_split_is_discarded():
    local = FakeLocalClient(answers=['["only one"]'])
    decomposer = Decomposer(DecomposerConfig(enabled=True), local)
    assert decomposer.split(LONG_TASK) == []


def test_route_decomposed_returns_none_when_not_applicable():
    local = FakeLocalClient(answers=["[]"])
    decomposer = Decomposer(DecomposerConfig(enabled=True), local)
    ladder = make_ladder(FakeLocalClient(), FakeRemoteClient())
    assert decomposer.route_decomposed(LONG_TASK, ladder) is None


def test_route_decomposed_routes_each_subtask_and_composes():
    # Split model: returns two subtasks, then composes with a final answer.
    split_local = FakeLocalClient(
        answers=['["cost of pens?", "cost of books?"]', "36 and 100"]
    )
    decomposer = Decomposer(DecomposerConfig(enabled=True), split_local)

    ladder_local = FakeLocalClient(answers=["Answer: 36"], logprob_mean=-0.05)
    ladder = make_ladder(ladder_local, FakeRemoteClient())

    result = decomposer.route_decomposed(LONG_TASK, ladder)

    assert result is not None
    assert result.answer == "36 and 100"
    assert result.remote_tokens == 0  # both subtasks resolved locally
    assert len(ladder_local.calls) == 2  # one route per subtask


def test_compose_falls_back_to_concatenation_on_local_failure():
    failing_local = FakeLocalClient(fail=True)
    decomposer = Decomposer(DecomposerConfig(enabled=True), failing_local)
    merged = decomposer.compose("task", [("s1", "a1"), ("s2", "a2")])
    assert merged == "a1\na2"
