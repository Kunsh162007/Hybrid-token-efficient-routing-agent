"""Configuration loading and validation.

Every launch-day-dependent value (model IDs, thresholds, budgets) lives in
config.yaml so nothing needs a code change when models are revealed.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator

DEFAULT_CONFIG_PATH = "config.yaml"
CONFIG_ENV_VAR = "ROUTING_AGENT_CONFIG"
API_KEY_ENV_VAR = "FIREWORKS_API_KEY"


class LocalModelConfig(BaseModel):
    enabled: bool = True
    model_path: str = "models/gemma-3-1b-it-q4_0.gguf"
    n_ctx: int = Field(default=4096, ge=512)
    n_threads: int = Field(default=0, ge=0)
    max_tokens: int = Field(default=512, ge=1)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    retry_temperature: float = Field(default=0.8, ge=0.0, le=2.0)


class RemoteModelConfig(BaseModel):
    base_url: str = "https://api.fireworks.ai/inference/v1"
    cheap_model: str = "accounts/fireworks/models/gemma-3-4b-it"
    strong_model: str = "accounts/fireworks/models/llama-v3p1-70b-instruct"
    judge_model: str = "accounts/fireworks/models/gemma-3-4b-it"
    max_tokens_cheap: int = Field(default=256, ge=1)
    max_tokens_strong: int = Field(default=512, ge=1)
    max_tokens_judge: int = Field(default=4, ge=1)
    timeout_seconds: float = Field(default=60.0, gt=0)
    max_retries: int = Field(default=3, ge=0)

    @field_validator("base_url")
    @classmethod
    def _no_trailing_slash(cls, v: str) -> str:
        return v.rstrip("/")


class LadderConfig(BaseModel):
    confidence_threshold: float = Field(default=0.72, ge=0.0, le=1.0)
    self_consistency_k: int = Field(default=5, ge=1)
    unanimous_ratio: float = Field(default=1.0, ge=0.5, le=1.0)
    contested_ratio: float = Field(default=0.6, ge=0.0, le=1.0)
    skip_ahead_difficulty: float = Field(default=0.85, ge=0.0, le=1.0)
    per_task_token_budget: int = Field(default=2000, ge=1)
    wall_clock_cap_seconds: float = Field(default=120.0, gt=0)
    judge_enabled: bool = True


class CacheConfig(BaseModel):
    enabled: bool = True
    db_path: str = "data/cache.db"
    semantic_threshold: float = Field(default=0.95, ge=0.0, le=1.0)
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"


class LearnedRouterConfig(BaseModel):
    enabled: bool = False
    model_path: str = "data/router_model.joblib"
    min_probability_local: float = Field(default=0.5, ge=0.0, le=1.0)
    training_log_path: str = "data/training_records.jsonl"


class DecomposerConfig(BaseModel):
    enabled: bool = False
    max_subtasks: int = Field(default=4, ge=2, le=8)
    max_depth: int = Field(default=1, ge=1, le=1)  # depth >1 is deliberately unsupported


class WebConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1, le=65535)
    demo_mode: bool = False
    demo_max_tokens: int = Field(default=128, ge=1)


class AppConfig(BaseModel):
    local: LocalModelConfig = LocalModelConfig()
    remote: RemoteModelConfig = RemoteModelConfig()
    ladder: LadderConfig = LadderConfig()
    cache: CacheConfig = CacheConfig()
    learned_router: LearnedRouterConfig = LearnedRouterConfig()
    decomposer: DecomposerConfig = DecomposerConfig()
    web: WebConfig = WebConfig()


class ConfigError(Exception):
    """Raised when configuration is missing or invalid."""


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load and validate config from YAML; fall back to defaults if absent."""
    resolved = Path(path or os.environ.get(CONFIG_ENV_VAR, DEFAULT_CONFIG_PATH))
    if not resolved.exists():
        if path is not None:
            raise ConfigError(f"Config file not found: {resolved}")
        return AppConfig()
    try:
        raw = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {resolved}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"Config root must be a mapping, got {type(raw).__name__}")
    return AppConfig.model_validate(raw)


def get_api_key() -> str:
    """Fetch the Fireworks API key from the environment; never from files."""
    key = os.environ.get(API_KEY_ENV_VAR, "").strip()
    if not key:
        raise ConfigError(
            f"{API_KEY_ENV_VAR} is not set. Copy .env.example to .env or export it."
        )
    return key
