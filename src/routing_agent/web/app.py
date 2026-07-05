"""FastAPI app: routing endpoint, live stats, and the dashboard page."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from routing_agent import __version__
from routing_agent.types import Rung

_STATIC_DIR = Path(__file__).parent / "static"


class RouteRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=20_000)


def create_app(runtime) -> FastAPI:
    app = FastAPI(title="Hybrid Token-Efficient Routing Agent", version=__version__)

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "version": __version__,
            "local_model": runtime.local_available,
            "remote": runtime.remote_available,
        }

    @app.post("/api/route")
    def route(request: RouteRequest):
        try:
            result = runtime.route_task(request.prompt)
        except Exception as exc:  # surface as a clean 502, never a stack trace
            raise HTTPException(status_code=502, detail=f"Routing failed: {exc}") from exc
        return {
            "answer": result.answer,
            "exit_rung": result.exit_rung.name,
            "exit_rung_number": int(result.exit_rung),
            "confidence": round(result.confidence, 3),
            "remote_tokens": result.remote_tokens,
            "task_type": str(result.task_type),
            "cached": result.cached,
            "verified": result.verified,
            "elapsed_seconds": round(result.elapsed_seconds, 2),
            "trace": [
                {
                    "rung": trace.rung.name,
                    "rung_number": int(trace.rung),
                    "action": trace.action,
                    "detail": trace.detail,
                    "remote_tokens": trace.remote_tokens,
                }
                for trace in result.trace
            ],
        }

    @app.get("/api/stats")
    def stats():
        snapshot = runtime.budget.snapshot()
        return {
            "tasks_completed": snapshot.tasks_completed,
            "remote_tokens_spent": snapshot.remote_tokens_spent,
            "local_tokens_used": snapshot.local_tokens_used,
            "free_task_ratio": round(snapshot.free_task_ratio, 3),
            "rung_exits": {
                Rung(rung).name: count for rung, count in snapshot.rung_exits.items()
            },
            "cache_hits": runtime.cache.hits if runtime.cache else 0,
            "cache_semantic_hits": runtime.cache.semantic_hits if runtime.cache else 0,
            "cache_size": runtime.cache.size() if runtime.cache else 0,
            "local_model": runtime.local_available,
            "remote": runtime.remote_available,
        }

    @app.get("/", response_class=HTMLResponse)
    def index():
        return (_STATIC_DIR / "index.html").read_text(encoding="utf-8")

    return app


def create_default_app() -> FastAPI:
    """uvicorn --factory entrypoint: builds the runtime from config/env."""
    from routing_agent.runtime import build_runtime

    return create_app(build_runtime())
