"""Local Gemma client backed by llama.cpp.

Local tokens are free under the scoring rules, so this client is used
aggressively. It degrades gracefully: if llama-cpp-python or the GGUF file is
missing, constructing the client raises LocalModelUnavailable and the ladder
falls back to remote-only mode instead of crashing.
"""

from __future__ import annotations

import math
from pathlib import Path

from routing_agent.clients.base import GenerationError, LocalModelUnavailable
from routing_agent.config import LocalModelConfig
from routing_agent.types import GenerationResult

_DEFAULT_SYSTEM = "You are a precise assistant. Answer correctly and concisely."


class LocalGemmaClient:
    """Wraps a GGUF Gemma model via llama-cpp-python with logprob capture."""

    def __init__(self, config: LocalModelConfig) -> None:
        if not config.enabled:
            raise LocalModelUnavailable("Local model disabled in config")
        model_path = Path(config.model_path)
        if not model_path.exists():
            raise LocalModelUnavailable(
                f"GGUF model not found at {model_path}. "
                "Download it first (see README) or set local.enabled: false."
            )
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise LocalModelUnavailable(
                "llama-cpp-python is not installed. Install with: pip install '.[local]'"
            ) from exc

        self._config = config
        self._llm = Llama(
            model_path=str(model_path),
            n_ctx=config.n_ctx,
            n_threads=config.n_threads or None,
            logits_all=True,
            verbose=False,
        )
        self.model_id = model_path.stem

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        system: str | None = None,
    ) -> GenerationResult:
        try:
            response = self._llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": system or _DEFAULT_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens or self._config.max_tokens,
                temperature=(
                    temperature if temperature is not None else self._config.temperature
                ),
                logprobs=True,
                top_logprobs=1,
            )
        except Exception as exc:
            raise GenerationError(f"Local generation failed: {exc}") from exc

        choice = response["choices"][0]
        text = (choice.get("message", {}).get("content") or "").strip()
        usage = response.get("usage", {})
        return GenerationResult(
            text=text,
            model_id=self.model_id,
            is_remote=False,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            logprob_mean=_mean_logprob(choice),
        )


def _mean_logprob(choice: dict) -> float | None:
    """Mean token logprob of the generated answer; the free confidence signal."""
    logprobs = choice.get("logprobs") or {}
    content = logprobs.get("content") or logprobs.get("token_logprobs")
    if not content:
        return None
    values: list[float] = []
    for entry in content:
        value = entry.get("logprob") if isinstance(entry, dict) else entry
        try:
            number = float(value)  # accepts numpy float32 from llama.cpp
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            values.append(number)
    if not values:
        return None
    return sum(values) / len(values)
