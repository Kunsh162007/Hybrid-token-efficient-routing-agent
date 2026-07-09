"""Fireworks AI client (OpenAI-compatible chat completions).

Every call here costs scored tokens, so prompts are expected to be
pre-compressed and max_tokens tightly capped by the caller.
"""

from __future__ import annotations

import time

import httpx

from routing_agent.clients.base import RemoteModelError
from routing_agent.config import RemoteModelConfig
from routing_agent.types import GenerationResult

_TERSE_SYSTEM = (
    "You are a precise assistant. Reply with only the final answer. "
    "No explanations, no preamble, no markdown unless asked."
)
_JUDGE_SYSTEM = (
    "You are a strict grader. Given a task and a candidate answer, "
    "reply with exactly one word: YES if the answer is correct, NO otherwise."
)
_RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}


class FireworksClient:
    """Minimal, retrying Fireworks chat-completions client."""

    def __init__(
        self,
        config: RemoteModelConfig,
        api_key: str,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._config = config
        self._client = httpx.Client(
            base_url=config.base_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=config.timeout_seconds,
            transport=transport,
        )

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        system: str | None = None,
        model: str | None = None,
    ) -> GenerationResult:
        model_id = model or self._config.cheap_model
        payload = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": system or _TERSE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_tokens or self._config.max_tokens_cheap,
            "temperature": temperature if temperature is not None else 0.0,
        }
        for key, value in self._config.extra_params.items():
            payload.setdefault(key, value)
        data = self._post_with_retries(payload)
        text = self._extract_text(data)
        usage = data.get("usage") or {}
        return GenerationResult(
            text=text,
            model_id=model_id,
            is_remote=True,
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
        )

    def judge(self, task: str, candidate: str) -> tuple[bool, GenerationResult]:
        """1-token verdict on a local candidate answer. The cheapest remote rung."""
        prompt = f"Task:\n{task}\n\nCandidate answer:\n{candidate}\n\nIs it correct?"
        result = self.generate(
            prompt,
            max_tokens=self._config.max_tokens_judge,
            system=_JUDGE_SYSTEM,
            model=self._config.judge_model,
        )
        # Default to NO (which escalates) unless the grader clearly says YES:
        # a truncated reasoning-model verdict must never be read as approval.
        head = result.text.strip().upper()
        verdict = head.startswith("YES") or head == "Y"
        return verdict, result

    @staticmethod
    def _extract_text(data: dict) -> str:
        """Pull the answer out of a chat-completion, tolerating reasoning models.

        gpt-oss / deepseek-r1 style models return their chain of thought in
        `reasoning_content` and may leave `content` null or absent entirely when
        the token budget truncates before the final channel. A missing key here
        used to raise and be treated as a remote failure - which silently
        collapsed the whole agent onto the weak local model. Never raise: an
        empty string just fails the verifier and climbs one more rung.
        """
        choices = data.get("choices") or []
        if not choices:
            return ""
        message = choices[0].get("message") or {}
        content = message.get("content")
        if content and content.strip():
            return content.strip()
        reasoning = message.get("reasoning_content")
        if reasoning and reasoning.strip():
            return reasoning.strip()
        return ""

    def _post_with_retries(self, payload: dict) -> dict:
        last_error: Exception | None = None
        for attempt in range(self._config.max_retries + 1):
            try:
                response = self._client.post("/chat/completions", json=payload)
                if response.status_code == 200:
                    return response.json()
                if response.status_code not in _RETRYABLE_STATUS:
                    raise RemoteModelError(
                        f"Fireworks API error {response.status_code}: {response.text[:200]}"
                    )
                last_error = RemoteModelError(
                    f"Fireworks API {response.status_code} (attempt {attempt + 1})"
                )
            except httpx.HTTPError as exc:
                last_error = RemoteModelError(f"Network error calling Fireworks: {exc}")
            if attempt < self._config.max_retries:
                time.sleep(min(2**attempt, 8))
        raise last_error or RemoteModelError("Fireworks call failed")

    def close(self) -> None:
        self._client.close()
