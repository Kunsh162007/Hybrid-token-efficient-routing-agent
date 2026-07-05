"""Runtime assembly: build the full agent from config, degrading gracefully.

Single place where clients, cache, thresholds, ladder, and decomposer are
wired together, shared by the CLI and the web app.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from routing_agent.budget import BudgetTracker
from routing_agent.cache import AnswerCache
from routing_agent.clients.base import LocalModelUnavailable
from routing_agent.clients.local import LocalGemmaClient
from routing_agent.clients.remote import FireworksClient
from routing_agent.config import AppConfig, ConfigError, get_api_key, load_config
from routing_agent.router.adaptive import AdaptiveThresholds
from routing_agent.router.decomposer import Decomposer
from routing_agent.router.ladder import EscalationLadder
from routing_agent.types import TaskResult

logger = logging.getLogger(__name__)


@dataclass
class Runtime:
    """Assembled agent plus the shared accounting objects."""

    config: AppConfig
    ladder: EscalationLadder
    budget: BudgetTracker
    cache: AnswerCache | None
    decomposer: Decomposer
    thresholds: AdaptiveThresholds
    local_available: bool
    remote_available: bool

    def route_task(self, prompt: str) -> TaskResult:
        """Route one task, trying decomposition first when enabled."""
        if self.decomposer.enabled:
            decomposed = self.decomposer.route_decomposed(prompt, self.ladder)
            if decomposed is not None:
                return decomposed
        return self.ladder.route(prompt)


def build_runtime(config_path: str | None = None) -> Runtime:
    config = load_config(config_path)
    if config.web.demo_mode:
        config = _apply_demo_caps(config)

    local_client = None
    try:
        local_client = LocalGemmaClient(config.local)
        logger.info("Local model loaded: %s", config.local.model_path)
    except LocalModelUnavailable as exc:
        logger.warning("Local model unavailable (%s); remote-only mode", exc)

    remote_client = None
    try:
        remote_client = FireworksClient(config.remote, get_api_key())
    except ConfigError as exc:
        logger.warning("Remote client unavailable (%s); local-only mode", exc)

    if local_client is None and remote_client is None:
        raise ConfigError(
            "Neither local model nor Fireworks API key available - nothing can "
            "answer tasks. Download the GGUF model or set FIREWORKS_API_KEY."
        )

    cache = None
    if config.cache.enabled:
        cache = AnswerCache(config.cache)
        logger.info(
            "Cache ready (semantic=%s, %d entries)", cache.semantic_enabled, cache.size()
        )

    thresholds = AdaptiveThresholds(config.ladder.confidence_threshold)
    budget = BudgetTracker(config.ladder.per_task_token_budget)

    estimator = _load_learned_estimator(config)

    ladder = EscalationLadder(
        config.ladder,
        config.local,
        config.remote,
        local_client,
        remote_client,
        budget,
        thresholds=thresholds,
        cache=cache,
        difficulty_estimator=estimator,
    )
    decomposer = Decomposer(config.decomposer, local_client)

    return Runtime(
        config=config,
        ladder=ladder,
        budget=budget,
        cache=cache,
        decomposer=decomposer,
        thresholds=thresholds,
        local_available=local_client is not None,
        remote_available=remote_client is not None,
    )


def _apply_demo_caps(config: AppConfig) -> AppConfig:
    """Demo mode: shorter generations so the dashboard stays snappy."""
    cap = config.web.demo_max_tokens
    return config.model_copy(
        update={
            "local": config.local.model_copy(
                update={"max_tokens": min(config.local.max_tokens, cap)}
            ),
            "remote": config.remote.model_copy(
                update={
                    "max_tokens_cheap": min(config.remote.max_tokens_cheap, cap),
                    "max_tokens_strong": min(config.remote.max_tokens_strong, cap),
                }
            ),
        }
    )


def _load_learned_estimator(config: AppConfig):
    if not config.learned_router.enabled:
        return None
    from routing_agent.router.learned import LearnedRouter, LearnedRouterUnavailable

    try:
        router = LearnedRouter.load(config.learned_router.model_path)
        logger.info("Learned router loaded from %s", config.learned_router.model_path)
        return router.as_estimator()
    except LearnedRouterUnavailable as exc:
        logger.warning("Learned router unavailable (%s); using heuristics", exc)
        return None
