"""Local Gemma client backed by llama.cpp.

Local tokens are free under the scoring rules, so this client is used
aggressively. It degrades gracefully: if llama-cpp-python or the GGUF file is
missing, constructing the client raises LocalModelUnavailable and the ladder
falls back to remote-only mode instead of crashing.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

from routing_agent.clients.base import GenerationError, LocalModelUnavailable
from routing_agent.config import LocalModelConfig
from routing_agent.types import GenerationResult

logger = logging.getLogger(__name__)

_DEFAULT_SYSTEM = (
    "You are a precise assistant. Answer correctly and concisely, in English."
)

_CGROUP_V2_CPU_MAX = "/sys/fs/cgroup/cpu.max"
_CGROUP_V1_QUOTA = "/sys/fs/cgroup/cpu/cpu.cfs_quota_us"
_CGROUP_V1_PERIOD = "/sys/fs/cgroup/cpu/cpu.cfs_period_us"


class LocalGemmaClient:
    """Wraps a GGUF Gemma model via llama-cpp-python."""

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
        n_threads = resolve_thread_count(config.n_threads)
        logger.info("Local model threads: %d", n_threads)
        try:
            # logits_all=False is a large prefill win: True makes llama.cpp
            # run the 262k-vocab LM head on EVERY prompt token. The cost is
            # losing logprob confidence (this llama-cpp-python raises if
            # logprobs are requested without it) - at the shipped 0.98
            # threshold that signal never gated anything, so the ladder's
            # verifier + self-consistency + judge carry trust instead.
            self._llm = Llama(
                model_path=str(model_path),
                n_ctx=config.n_ctx,
                n_threads=n_threads,
                logits_all=False,
                verbose=False,
            )
        except Exception as exc:  # corrupt GGUF, mmap/alloc failure, ABI drift
            raise LocalModelUnavailable(
                f"llama.cpp failed to load {model_path}: {exc}"
            ) from exc
        self.model_id = model_path.stem

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        system: str | None = None,
        deadline: float | None = None,
    ) -> GenerationResult:
        """Generate one completion, abandoning it if `deadline` passes.

        `deadline` is a `time.monotonic()` timestamp. llama.cpp offers no
        wall-clock cap and a CPU generation is otherwise uninterruptible, so
        the token stream is consumed incrementally and dropped the moment the
        deadline is crossed. A truncated answer is worse than no answer here -
        the verifier can wave through a half-finished prose answer - so an
        overrun raises GenerationError and the ladder escalates instead.
        """
        if deadline is not None and time.monotonic() >= deadline:
            raise GenerationError("Local generation skipped: deadline already passed")

        try:
            stream = self._llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": system or _DEFAULT_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens or self._config.max_tokens,
                temperature=(
                    temperature if temperature is not None else self._config.temperature
                ),
                stream=True,
            )
        except Exception as exc:
            raise GenerationError(f"Local generation failed: {exc}") from exc

        pieces: list[str] = []
        completion_tokens = 0
        try:
            for chunk in stream:
                if deadline is not None and time.monotonic() >= deadline:
                    raise GenerationError(
                        f"Local generation exceeded its time budget after "
                        f"{completion_tokens} tokens"
                    )
                delta = chunk["choices"][0].get("delta") or {}
                piece = delta.get("content")
                if piece:
                    pieces.append(piece)
                    completion_tokens += 1
        except GenerationError:
            raise
        except Exception as exc:
            raise GenerationError(f"Local generation failed: {exc}") from exc
        finally:
            # Closing the generator releases llama.cpp's sampling loop; without
            # it an abandoned stream keeps the context pinned for the next call.
            close = getattr(stream, "close", None)
            if close is not None:
                close()

        return GenerationResult(
            text="".join(pieces).strip(),
            model_id=self.model_id,
            is_remote=False,
            prompt_tokens=self._count_prompt_tokens(prompt, system),
            completion_tokens=completion_tokens,
            # Streamed chunks carry no logprobs, and logits_all=False meant none
            # were available on the buffered path either. Confidence stays a
            # neutral 0.5 and trust comes from verifier + quorum + judge.
            logprob_mean=None,
        )

    def _count_prompt_tokens(self, prompt: str, system: str | None) -> int:
        """Best-effort prompt size; streamed responses carry no usage block.

        Local tokens are never billed, so this is telemetry only - a tokenizer
        hiccup must not fail a generation that already succeeded.
        """
        try:
            text = f"{system or _DEFAULT_SYSTEM}\n{prompt}"
            return len(self._llm.tokenize(text.encode("utf-8")))
        except Exception:
            return 0


def resolve_thread_count(configured: int) -> int:
    """Threads for llama.cpp: explicit config, else the real CPU allowance.

    llama.cpp's own auto-detect reads the *host* processor count and cannot see
    a cgroup CPU quota. On the graded 2-vCPU container that means spawning one
    thread per host core while the cgroup throttles the whole group to 2
    CPU-seconds per second - pure context-switch thrash. Detect the quota
    ourselves and pin the thread count to it.
    """
    if configured > 0:
        return configured
    limits = [n for n in (_cgroup_cpu_quota(), _affinity_cpus()) if n]
    return max(1, min(limits)) if limits else 1


def _cgroup_cpu_quota() -> int | None:
    """CPU count implied by the cgroup quota, or None when unlimited/absent."""
    quota = _read_cgroup_v2_quota()
    if quota is None:
        quota = _read_cgroup_v1_quota()
    if quota is None:
        return None
    # Floor, never below 1: a 1.5-CPU quota runs better on one thread than on
    # two threads fighting over it.
    return max(1, int(quota))


def _read_cgroup_v2_quota() -> float | None:
    """Parse `/sys/fs/cgroup/cpu.max`, formatted as "<quota|max> <period>"."""
    try:
        fields = Path(_CGROUP_V2_CPU_MAX).read_text(encoding="utf-8").split()
    except OSError:
        return None
    if len(fields) != 2 or fields[0] == "max":
        return None
    try:
        quota, period = float(fields[0]), float(fields[1])
    except ValueError:
        return None
    return quota / period if period > 0 else None


def _read_cgroup_v1_quota() -> float | None:
    """Parse cgroup v1's cfs quota/period pair; quota -1 means unlimited."""
    try:
        quota = float(Path(_CGROUP_V1_QUOTA).read_text(encoding="utf-8").strip())
        period = float(Path(_CGROUP_V1_PERIOD).read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None
    if quota <= 0 or period <= 0:
        return None
    return quota / period


def _affinity_cpus() -> int | None:
    """CPUs this process may actually run on (honours `--cpuset-cpus`)."""
    getaffinity = getattr(os, "sched_getaffinity", None)  # Linux only
    if getaffinity is not None:
        try:
            return len(getaffinity(0))
        except OSError:
            pass
    return os.cpu_count()
