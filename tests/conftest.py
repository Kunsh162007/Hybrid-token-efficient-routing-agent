"""Shared test fakes: scriptable local and remote clients."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from routing_agent.types import GenerationResult


@dataclass
class FakeLocalClient:
    """Scriptable local client: returns queued answers, then repeats the last."""

    answers: list[str] = field(default_factory=lambda: ["42"])
    logprob_mean: float = -0.1
    fail: bool = False
    model_id: str = "fake-gemma"
    calls: list[dict] = field(default_factory=list)

    def generate(self, prompt, *, max_tokens=None, temperature=None, system=None):
        self.calls.append(
            {"prompt": prompt, "max_tokens": max_tokens, "temperature": temperature}
        )
        if self.fail:
            from routing_agent.clients.base import GenerationError

            raise GenerationError("scripted local failure")
        index = min(len(self.calls) - 1, len(self.answers) - 1)
        return GenerationResult(
            text=self.answers[index],
            model_id=self.model_id,
            is_remote=False,
            prompt_tokens=len(prompt.split()),
            completion_tokens=len(self.answers[index].split()),
            logprob_mean=self.logprob_mean,
        )


@dataclass
class FakeRemoteClient:
    """Scriptable remote client with a controllable judge verdict."""

    answer: str = "remote-answer"
    judge_verdict: bool = True
    prompt_tokens: int = 50
    completion_tokens: int = 10
    calls: list[dict] = field(default_factory=list)
    judge_calls: list[dict] = field(default_factory=list)

    def generate(self, prompt, *, max_tokens=None, temperature=None, system=None, model=None):
        self.calls.append({"prompt": prompt, "max_tokens": max_tokens, "model": model})
        return GenerationResult(
            text=self.answer,
            model_id=model or "fake-fireworks",
            is_remote=True,
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
        )

    def judge(self, task, candidate):
        self.judge_calls.append({"task": task, "candidate": candidate})
        result = GenerationResult(
            text="YES" if self.judge_verdict else "NO",
            model_id="fake-judge",
            is_remote=True,
            prompt_tokens=30,
            completion_tokens=1,
        )
        return self.judge_verdict, result


@pytest.fixture
def fake_local():
    return FakeLocalClient()


@pytest.fixture
def fake_remote():
    return FakeRemoteClient()
