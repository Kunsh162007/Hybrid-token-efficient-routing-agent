"""Escalation ladder behavior tests using scripted fake clients."""

from conftest import FakeLocalClient, FakeRemoteClient

from routing_agent.budget import BudgetTracker
from routing_agent.config import LadderConfig, LocalModelConfig, RemoteModelConfig
from routing_agent.router.ladder import EscalationLadder
from routing_agent.types import Classification, Rung, TaskType

MATH_PROMPT = "What is 2+2?"


def make_ladder(local, remote, *, budget=None, cache=None, estimator=None, **ladder_kwargs):
    return EscalationLadder(
        LadderConfig(**ladder_kwargs),
        LocalModelConfig(),
        RemoteModelConfig(),
        local,
        remote,
        budget or BudgetTracker(per_task_budget=2000),
        cache=cache,
        difficulty_estimator=estimator,
    )


def test_confident_local_answer_exits_rung_1_free():
    local = FakeLocalClient(answers=["Answer: 4"], logprob_mean=-0.05)
    remote = FakeRemoteClient()

    result = make_ladder(local, remote).route(MATH_PROMPT)

    assert result.exit_rung == Rung.LOCAL_FIRST
    assert result.remote_tokens == 0 and result.was_free
    assert result.verified is True
    assert len(local.calls) == 1
    assert remote.calls == [] and remote.judge_calls == []


def test_unconfident_but_unanimous_vote_ships_free_at_rung_3():
    # Confidence ~0.14 is below threshold, but samples agree unanimously:
    # the early-consensus quorum (3) stops sampling before k (5).
    local = FakeLocalClient(answers=["Answer: 4"], logprob_mean=-2.0)
    remote = FakeRemoteClient()

    result = make_ladder(local, remote).route(MATH_PROMPT)

    assert result.exit_rung == Rung.SELF_CONSISTENCY
    assert result.remote_tokens == 0
    assert len(local.calls) == 3  # quorum, not k
    assert remote.calls == []
    assert any(t.action == "early-consensus" for t in result.trace)


def test_dissent_disables_early_consensus_and_samples_to_k():
    # One dissenting answer forces the full k samples; 4/5 agreement is
    # contested-not-unanimous, so the winner goes to the remote judge.
    local = FakeLocalClient(
        answers=["Answer: 4", "Answer: 5", "Answer: 4", "Answer: 4", "Answer: 4"],
        logprob_mean=-2.0,
    )
    remote = FakeRemoteClient(judge_verdict=True)

    result = make_ladder(local, remote).route(MATH_PROMPT)

    assert len(local.calls) == 5  # dissent at sample 2 -> full k
    assert result.exit_rung == Rung.REMOTE_JUDGE
    assert result.remote_tokens == 31  # judge only, no full generation
    assert "4" in result.answer


def test_confident_code_answer_still_asks_the_judge():
    # Syntax-valid buggy code passes the free verifier, so CODE never ships
    # on rung-1 confidence alone: the 1-token judge must say YES first.
    code = "```python\ndef add(a, b):\n    return a + b\n```"
    local = FakeLocalClient(answers=[code], logprob_mean=-0.05)
    remote = FakeRemoteClient(judge_verdict=True)

    result = make_ladder(local, remote).route("Fix the bug in this code: ...")

    assert result.exit_rung == Rung.REMOTE_JUDGE
    assert len(remote.judge_calls) == 1
    assert remote.calls == []  # no full paid generation


def test_code_judge_no_escalates_instead_of_shipping():
    code = "```python\ndef add(a, b):\n    return a - b\n```"
    local = FakeLocalClient(answers=[code], logprob_mean=-0.05)
    remote = FakeRemoteClient(judge_verdict=False, answer="def add(a, b):\n    return a + b")

    result = make_ladder(local, remote).route("Fix the bug in this python code: ...")

    assert result.exit_rung >= Rung.REMOTE_CHEAP
    assert len(remote.calls) >= 1


def test_code_ships_locally_when_no_remote_available():
    # Local-only degraded mode: the judge gate must not block shipping.
    code = "```python\ndef add(a, b):\n    return a + b\n```"
    local = FakeLocalClient(answers=[code], logprob_mean=-0.05)

    result = make_ladder(local, None).route("Fix the bug in this code: ...")

    assert result.exit_rung == Rung.LOCAL_FIRST
    assert result.was_free


def test_per_type_max_tokens_cap_is_used():
    local = FakeLocalClient(answers=["Answer: Tokyo"], logprob_mean=-0.05)
    ladder = EscalationLadder(
        LadderConfig(),
        LocalModelConfig(max_tokens=512, max_tokens_by_type={"qa": 96}),
        RemoteModelConfig(),
        local,
        FakeRemoteClient(),
        BudgetTracker(per_task_budget=2000),
    )

    ladder.route("What is the capital of Japan?")

    assert local.calls[0]["max_tokens"] == 96


def test_contested_vote_goes_to_judge_and_ships_on_yes():
    local = FakeLocalClient(
        answers=["Answer: 4", "Answer: 5", "Answer: 4", "Answer: 5", "Answer: 4"],
        logprob_mean=-2.0,
    )
    remote = FakeRemoteClient(judge_verdict=True)

    result = make_ladder(local, remote).route(MATH_PROMPT)

    assert result.exit_rung == Rung.REMOTE_JUDGE
    assert result.remote_tokens == 31  # judge call only (30 + 1)
    assert "4" in result.answer
    assert len(remote.judge_calls) == 1
    assert remote.calls == []  # no full remote generation


def test_judge_no_escalates_to_cheap_remote():
    local = FakeLocalClient(
        answers=["Answer: 4", "Answer: 5", "Answer: 4", "Answer: 5", "Answer: 4"],
        logprob_mean=-2.0,
    )
    remote = FakeRemoteClient(judge_verdict=False, answer="7")

    result = make_ladder(local, remote).route(MATH_PROMPT)

    assert result.exit_rung == Rung.REMOTE_CHEAP
    assert result.answer == "7"
    assert result.remote_tokens == 31 + 60  # judge + cheap generation
    assert len(remote.calls) == 1


def test_no_local_model_goes_straight_to_remote():
    remote = FakeRemoteClient(answer="4")

    result = make_ladder(None, remote).route(MATH_PROMPT)

    assert result.exit_rung == Rung.REMOTE_CHEAP
    assert result.remote_tokens == 60
    assert any(t.action == "skip-local" for t in result.trace)


def test_skip_ahead_on_high_difficulty():
    local = FakeLocalClient(answers=["Answer: 4"])
    remote = FakeRemoteClient(answer="4")
    estimator = lambda prompt: Classification(TaskType.QA, difficulty=0.95)  # noqa: E731

    result = make_ladder(local, remote, estimator=estimator).route(MATH_PROMPT)

    assert result.exit_rung == Rung.REMOTE_CHEAP
    assert local.calls == []  # local never attempted


def test_budget_exhaustion_settles_for_best_local_candidate():
    local = FakeLocalClient(
        answers=["Answer: 4", "Answer: 5", "Answer: 4", "Answer: 5", "Answer: 4"],
        logprob_mean=-2.0,
    )
    remote = FakeRemoteClient(judge_verdict=False)
    budget = BudgetTracker(per_task_budget=10)  # too small for any remote call

    result = make_ladder(local, remote, budget=budget).route(MATH_PROMPT)

    assert result.exit_rung == Rung.SELF_CONSISTENCY
    assert result.remote_tokens == 0
    assert result.answer  # best local candidate, not empty
    assert remote.calls == [] and remote.judge_calls == []


def test_local_failure_falls_through_to_remote():
    local = FakeLocalClient(fail=True)
    remote = FakeRemoteClient(answer="4")

    result = make_ladder(local, remote).route(MATH_PROMPT)

    assert result.exit_rung == Rung.REMOTE_CHEAP
    assert result.answer == "4"


def test_cache_hit_returns_without_any_generation():
    class FakeCache:
        def __init__(self):
            self.put_calls = []

        def lookup(self, prompt):
            return "4"

        def put(self, prompt, answer):
            self.put_calls.append((prompt, answer))

    local = FakeLocalClient()
    remote = FakeRemoteClient()
    cache = FakeCache()

    result = make_ladder(local, remote, cache=cache).route(MATH_PROMPT)

    assert result.cached is True
    assert result.remote_tokens == 0
    assert local.calls == [] and remote.calls == []


def test_paid_remote_answer_is_stored_in_cache():
    class FakeCache:
        def __init__(self):
            self.put_calls = []

        def lookup(self, prompt):
            return None

        def put(self, prompt, answer):
            self.put_calls.append((prompt, answer))

    remote = FakeRemoteClient(answer="4")
    cache = FakeCache()

    make_ladder(None, remote, cache=cache).route(MATH_PROMPT)

    assert cache.put_calls == [(MATH_PROMPT, "4")]


def test_remote_failure_returns_best_effort_not_crash():
    from routing_agent.clients.base import GenerationError

    class BrokenRemote(FakeRemoteClient):
        def generate(self, *args, **kwargs):
            raise GenerationError("api down")

        def judge(self, *args, **kwargs):
            raise GenerationError("api down")

    local = FakeLocalClient(
        answers=["Answer: 4", "Answer: 5", "Answer: 4", "Answer: 5", "Answer: 4"],
        logprob_mean=-2.0,
    )

    result = make_ladder(local, BrokenRemote()).route(MATH_PROMPT)

    assert result.answer  # settles on a local candidate
    assert result.remote_tokens == 0
