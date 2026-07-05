"""Web API tests with an injected fake runtime."""

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from routing_agent.budget import BudgetTracker
from routing_agent.types import Rung, RungTrace, TaskResult, TaskType
from routing_agent.web.app import create_app


class FakeRuntime:
    def __init__(self):
        self.budget = BudgetTracker(per_task_budget=2000)
        self.cache = None
        self.local_available = True
        self.remote_available = True
        self.fail = False

    def route_task(self, prompt):
        if self.fail:
            raise RuntimeError("boom")
        return TaskResult(
            answer="4",
            exit_rung=Rung.LOCAL_FIRST,
            confidence=0.95,
            remote_tokens=0,
            task_type=TaskType.MATH,
            verified=True,
            elapsed_seconds=0.12,
            trace=(RungTrace(Rung.LOCAL_FIRST, "local-attempt", "verified=True"),),
        )


@pytest.fixture
def client():
    return TestClient(create_app(FakeRuntime()))


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok" and body["local_model"] is True


def test_route_returns_result_payload(client):
    response = client.post("/api/route", json={"prompt": "What is 2+2?"})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "4"
    assert body["exit_rung"] == "LOCAL_FIRST"
    assert body["remote_tokens"] == 0
    assert body["trace"][0]["action"] == "local-attempt"


def test_route_validates_empty_prompt(client):
    response = client.post("/api/route", json={"prompt": ""})
    assert response.status_code == 422


def test_route_failure_maps_to_502():
    runtime = FakeRuntime()
    runtime.fail = True
    client = TestClient(create_app(runtime))
    response = client.post("/api/route", json={"prompt": "x"})
    assert response.status_code == 502


def test_stats_shape(client):
    response = client.get("/api/stats")
    assert response.status_code == 200
    body = response.json()
    assert {"tasks_completed", "remote_tokens_spent", "free_task_ratio",
            "rung_exits", "cache_hits"} <= set(body)


def test_index_serves_dashboard(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "<html" in response.text.lower() or "<!doctype" in response.text.lower()
