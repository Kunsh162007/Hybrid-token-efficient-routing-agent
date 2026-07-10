"""Configuration loading and validation.

Every launch-day-dependent value (model IDs, thresholds, budgets) lives in
config.yaml so nothing needs a code change when models are revealed. The
judging harness injects FIREWORKS_BASE_URL and ALLOWED_MODELS at runtime;
those environment variables always override the YAML (hackathon rule: calls
that bypass the injected base URL or use non-allowed models score zero).
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = "config.yaml"
CONFIG_ENV_VAR = "ROUTING_AGENT_CONFIG"
API_KEY_ENV_VAR = "FIREWORKS_API_KEY"
BASE_URL_ENV_VAR = "FIREWORKS_BASE_URL"
ALLOWED_MODELS_ENV_VAR = "ALLOWED_MODELS"
APP_ROOT_ENV_VAR = "APP_ROOT"
# Where the image installs config.yaml and models/ (the Dockerfile WORKDIR).
_IMAGE_APP_ROOT = "/app"


def resolve_resource(relative: str | Path) -> Path:
    """Locate a bundled resource without depending on the working directory.

    config.yaml and the GGUF are named by relative paths, which resolve
    against the *process* CWD. A judging harness may start the container with
    any working directory, and when it does both files vanish at once:
    load_config() silently falls back to defaults and the local client reports
    the model missing. With no FIREWORKS_API_KEY injected, that pair raises
    ConfigError and every task ships an empty answer - a scored zero, produced
    without a single ERROR log.

    Relative paths are therefore searched against the CWD (dev boxes), then
    $APP_ROOT, then the image WORKDIR. Absolute paths pass through untouched.
    When nothing matches, the CWD-relative path is returned so failure
    messages still name the path the caller asked for.
    """
    path = Path(relative)
    if path.is_absolute():
        return path
    app_root = os.environ.get(APP_ROOT_ENV_VAR, "").strip() or "."
    for root in (Path.cwd(), Path(app_root), Path(_IMAGE_APP_ROOT)):
        candidate = root / path
        if candidate.exists():
            return candidate
    return path


class LocalModelConfig(BaseModel):
    enabled: bool = True
    model_path: str = "models/gemma-3-1b-it-q4_0.gguf"
    n_ctx: int = Field(default=4096, ge=512)
    n_threads: int = Field(default=0, ge=0)
    max_tokens: int = Field(default=512, ge=1)
    # Optional per-task-type caps (keys are TaskType values, e.g. "qa": 96).
    # Short-answer types finish sooner; unlisted types use max_tokens.
    max_tokens_by_type: dict[str, int] = Field(default_factory=dict)
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
    # Provider-specific request fields merged into every payload, e.g.
    # {"reasoning_effort": "low"} to stop gpt-oss burning hidden tokens.
    extra_params: dict = Field(default_factory=dict)

    @field_validator("base_url")
    @classmethod
    def _no_trailing_slash(cls, v: str) -> str:
        return v.rstrip("/")


class LadderConfig(BaseModel):
    confidence_threshold: float = Field(default=0.72, ge=0.0, le=1.0)
    self_consistency_k: int = Field(default=5, ge=1)
    # Stop sampling early once this many verified answers agree with zero
    # dissent; any disagreement falls back to the full k-sample vote.
    early_consensus_quorum: int = Field(default=3, ge=2)
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
    resolved = resolve_resource(path or os.environ.get(CONFIG_ENV_VAR, DEFAULT_CONFIG_PATH))
    if not resolved.exists():
        if path is not None:
            raise ConfigError(f"Config file not found: {resolved}")
        # Defaults are *not* the tuned values (confidence_threshold, the
        # per-type token caps): running on them quietly costs accuracy, so say
        # so rather than letting a missing file look like a healthy start.
        logger.warning(
            "No config file at %s (cwd=%s); falling back to built-in defaults",
            resolved, Path.cwd(),
        )
        return _apply_env_overrides(AppConfig())
    try:
        raw = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {resolved}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"Config root must be a mapping, got {type(raw).__name__}")
    return _apply_env_overrides(AppConfig.model_validate(raw))


def _apply_env_overrides(config: AppConfig) -> AppConfig:
    """Harness-injected env vars beat the YAML: base URL and allowed models."""
    updates: dict[str, str] = {}

    base_url = os.environ.get(BASE_URL_ENV_VAR, "").strip()
    if base_url:
        # model_copy(update=...) skips field validators, so the
        # _no_trailing_slash rule is applied by hand here - keep in sync.
        updates["base_url"] = base_url.rstrip("/")

    allowed = [
        model.strip()
        for model in os.environ.get(ALLOWED_MODELS_ENV_VAR, "").split(",")
        if model.strip()
    ]
    if allowed:
        cheap, strong = _pick_model_tiers(allowed)
        updates["cheap_model"] = cheap
        updates["strong_model"] = strong
        # 1-token verdicts don't need the strong model; judge on the cheap tier.
        updates["judge_model"] = cheap

    if not updates:
        return config
    return config.model_copy(
        update={"remote": config.remote.model_copy(update=updates)}
    )


_PARAM_COUNT = re.compile(r"(\d+(?:\.\d+)?)\s*b\b", re.IGNORECASE)
_MOE_COUNT = re.compile(r"(\d+)x(\d+(?:\.\d+)?)b\b", re.IGNORECASE)


def _model_size_billions(model_id: str) -> float | None:
    """Best-effort parameter count parsed from a model ID, in billions."""
    moe = _MOE_COUNT.search(model_id)
    if moe:
        return float(moe.group(1)) * float(moe.group(2))
    match = _PARAM_COUNT.search(model_id)
    return float(match.group(1)) if match else None


def _pick_model_tiers(allowed: list[str]) -> tuple[str, str]:
    """(cheap, strong) from the allowed list by parameter-size hint.

    IDs without a parseable size (deepseek-v3, kimi-k2, ...) are almost
    always flagship-large models, so unknown sizes rank as infinitely large
    rather than falling back to list order - otherwise a sized small model
    next to an unsized giant would invert the tiers. If ranking cannot
    separate the models, list order decides; a single allowed model fills
    both tiers.
    """
    if len(allowed) == 1:
        return allowed[0], allowed[0]

    def rank(model: str) -> float:
        size = _model_size_billions(model)
        return size if size is not None else float("inf")

    cheap = min(allowed, key=rank)
    strong = max(allowed, key=rank)
    if cheap == strong:
        return allowed[0], allowed[-1]
    return cheap, strong


def get_api_key() -> str:
    """Fetch the Fireworks API key from the environment (seeded from .env)."""
    key = os.environ.get(API_KEY_ENV_VAR, "").strip()
    if not key:
        _load_dotenv()
        key = os.environ.get(API_KEY_ENV_VAR, "").strip()
    if not key:
        raise ConfigError(
            f"{API_KEY_ENV_VAR} is not set. Copy .env.example to .env or export it."
        )
    return key


def _load_dotenv(path: str | Path = ".env") -> None:
    """Tiny .env loader: KEY=value lines, no expansion, env always wins."""
    env_file = Path(path)
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        name, _, value = stripped.partition("=")
        name = name.strip()
        if name and name not in os.environ:
            os.environ[name] = value.strip().strip("'\"")
