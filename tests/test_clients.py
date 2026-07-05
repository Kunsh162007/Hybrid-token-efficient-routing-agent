"""Fireworks client tests over a mock HTTP transport, plus local degradation."""

import httpx
import pytest

from routing_agent.clients.base import LocalModelUnavailable, RemoteModelError
from routing_agent.clients.local import LocalGemmaClient
from routing_agent.clients.remote import FireworksClient
from routing_agent.config import LocalModelConfig, RemoteModelConfig


def _make_client(handler) -> FireworksClient:
    config = RemoteModelConfig(max_retries=1, timeout_seconds=5)
    return FireworksClient(
        config, api_key="fw_test", transport=httpx.MockTransport(handler)
    )


def _ok_response(content="4", prompt_tokens=12, completion_tokens=3):
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": content}}],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            },
        },
    )


def test_generate_extracts_text_and_usage():
    # Arrange
    client = _make_client(lambda request: _ok_response("  4  "))

    # Act
    result = client.generate("2+2?")

    # Assert
    assert result.text == "4"
    assert result.is_remote is True
    assert result.billed_tokens == 15


def test_generate_sends_auth_and_caps_tokens():
    captured = {}

    def handler(request):
        captured["auth"] = request.headers["Authorization"]
        import json

        captured["payload"] = json.loads(request.content)
        return _ok_response()

    client = _make_client(handler)
    client.generate("hi", max_tokens=7)

    assert captured["auth"] == "Bearer fw_test"
    assert captured["payload"]["max_tokens"] == 7


def test_non_retryable_error_raises_immediately():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(401, text="bad key")

    client = _make_client(handler)
    with pytest.raises(RemoteModelError, match="401"):
        client.generate("hi")
    assert calls["n"] == 1


def test_retryable_error_retries_then_succeeds(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, text="rate limited")
        return _ok_response()

    client = _make_client(handler)
    result = client.generate("hi")
    assert result.text == "4"
    assert calls["n"] == 2


def test_judge_returns_yes_verdict_and_tokens():
    client = _make_client(lambda request: _ok_response("YES"))
    verdict, result = client.judge("2+2?", "4")
    assert verdict is True
    assert result.billed_tokens == 15


def test_judge_no_verdict():
    client = _make_client(lambda request: _ok_response("NO."))
    verdict, _ = client.judge("2+2?", "5")
    assert verdict is False


def test_malformed_response_raises():
    client = _make_client(lambda request: httpx.Response(200, json={"weird": True}))
    with pytest.raises(RemoteModelError, match="Malformed"):
        client.generate("hi")


def test_local_client_missing_weights_degrades(tmp_path):
    config = LocalModelConfig(model_path=str(tmp_path / "missing.gguf"))
    with pytest.raises(LocalModelUnavailable, match="not found"):
        LocalGemmaClient(config)


def test_local_client_disabled_degrades():
    config = LocalModelConfig(enabled=False)
    with pytest.raises(LocalModelUnavailable, match="disabled"):
        LocalGemmaClient(config)
