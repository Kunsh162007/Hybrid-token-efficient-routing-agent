"""Shared client protocol and error types."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from routing_agent.types import GenerationResult


class GenerationError(Exception):
    """A model call failed after all retries."""


class LocalModelUnavailable(GenerationError):
    """llama.cpp or the GGUF weights are not available in this environment."""


class RemoteModelError(GenerationError):
    """The Fireworks API returned an unrecoverable error."""


@runtime_checkable
class ModelClient(Protocol):
    """Uniform generation interface over local and remote backends."""

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        system: str | None = None,
    ) -> GenerationResult: ...
