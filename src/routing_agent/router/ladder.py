"""The escalation ladder: the token-spending policy at the heart of the agent.

Each rung costs more than the last; the loop exits at the first rung whose
answer can be trusted. Remote tokens are only spent from rung 4 onward.

    0  classify (heuristics, free)
    1  local attempt + verify          (free)
    2  local retry, hotter + reworded  (free)
    3  self-consistency majority vote  (free)
    4  remote judge, 1-token verdict   (cheapest paid rung)
    5  cheap remote, compressed prompt (paid)
    6  strong remote                   (most expensive, last resort)
"""

from __future__ import annotations

import time

from routing_agent.budget import BudgetExceeded, BudgetTracker
from routing_agent.clients.base import GenerationError
from routing_agent.config import LadderConfig, LocalModelConfig, RemoteModelConfig
from routing_agent.router import classifier as classifier_module
from routing_agent.router.adaptive import AdaptiveThresholds
from routing_agent.router.compression import compress_prompt
from routing_agent.router.confidence import logprob_to_confidence
from routing_agent.router.toolsolve import try_solve_math
from routing_agent.router.verifier import majority_vote, normalize, verify
from routing_agent.types import (
    Classification,
    GenerationResult,
    Rung,
    RungTrace,
    TaskResult,
    TaskType,
)

_LOCAL_INSTRUCTIONS: dict[TaskType, str] = {
    TaskType.MATH: "Think step by step briefly, then end with 'Answer: <number>'.",
    TaskType.MCQ: "Pick the best option. End with 'Answer: <letter>'.",
    TaskType.CODE: "Reply with only the code in a fenced block.",
    TaskType.EXTRACTION: (
        "Reply with only the extracted items, one per line. When entity "
        "types are requested, prefix each line with its type, e.g. "
        "'PERSON: ...', 'ORG: ...', 'LOCATION: ...', 'DATE: ...'."
    ),
    TaskType.SUMMARY: "Reply with only the summary.",
    TaskType.SENTIMENT: (
        "State the sentiment as one word (positive, negative, or neutral), "
        "then justify it in one short sentence."
    ),
    TaskType.LOGIC: (
        "Work through the constraints step by step, then end with "
        "'Answer: <answer>'."
    ),
    TaskType.QA: "Reply with only the answer. End with 'Answer: <answer>'.",
    TaskType.GENERAL: "Reply concisely with only what was asked.",
}
# Appended (not prepended) so rungs 1-3 share an identical prompt prefix and
# llama.cpp reuses the KV cache instead of reprocessing the whole prompt.
_RETRY_SUFFIX = "\n\nRe-read the task carefully before answering."
_DRAFT_HINT_MAX_CHARS = 200
_JUDGE_ESTIMATED_TOKENS = 150  # conservative pre-flight estimate for budget check
# Types where the free verifier is weakest: syntax-valid buggy code, fluent
# wrong deductions, and word problems whose numeric answer parses but is
# simply wrong. No local exit ships for these without a 1-token remote judge
# verdict - including unanimous self-consistency, because a 1B model agreeing
# with itself is not evidence (dry-run 2026-07-07: it unanimously returned
# wrong change on a money word problem and the buggy code unchanged).
_JUDGE_REQUIRED_TYPES = frozenset({TaskType.MATH, TaskType.CODE, TaskType.LOGIC})
# A local attempt below this time cap cannot finish on a slow CPU; go remote.
_MIN_LOCAL_ATTEMPT_SECONDS = 8.0


class _Candidate:
    """A local answer with its quality signals (internal bookkeeping)."""

    __slots__ = ("text", "confidence", "verified", "normalized")

    def __init__(self, text: str, confidence: float, verified: bool, normalized: str):
        self.text = text
        self.confidence = confidence
        self.verified = verified
        self.normalized = normalized


class EscalationLadder:
    """Routes one task through the cost-ordered rungs."""

    def __init__(
        self,
        ladder_config: LadderConfig,
        local_config: LocalModelConfig,
        remote_config: RemoteModelConfig,
        local_client,
        remote_client,
        budget: BudgetTracker,
        *,
        thresholds: AdaptiveThresholds | None = None,
        cache=None,
        difficulty_estimator=None,
    ) -> None:
        self._cfg = ladder_config
        self._local_cfg = local_config
        self._remote_cfg = remote_config
        self._local = local_client  # None => remote-only degraded mode
        self._remote = remote_client
        self._budget = budget
        self._thresholds = thresholds or AdaptiveThresholds(ladder_config.confidence_threshold)
        self._cache = cache
        # Pluggable rung-0 difficulty source (the learned router slots in here).
        self._estimate = difficulty_estimator or (lambda prompt: classifier_module.classify(prompt))

    def route(self, prompt: str, *, time_cap_seconds: float | None = None) -> TaskResult:
        """Route one task. `time_cap_seconds` overrides the config wall-clock
        cap for this call only (the batch harness shrinks it as the global
        deadline approaches). Threaded as a parameter, not instance state:
        the ladder is shared across dashboard threads."""
        time_cap = (
            time_cap_seconds
            if time_cap_seconds is not None
            else self._cfg.wall_clock_cap_seconds
        )
        started = time.monotonic()
        self._budget.begin_task()
        trace: list[RungTrace] = []

        cls: Classification = self._estimate(prompt)
        trace.append(
            RungTrace(
                Rung.CLASSIFY,
                "classified",
                f"type={cls.task_type} difficulty={cls.difficulty:.2f}",
            )
        )

        if self._cache is not None:
            cached = self._cache.lookup(prompt)
            if cached is not None:
                check = verify(cls.task_type, prompt, cached)
                if check.ok:
                    trace.append(RungTrace(Rung.CLASSIFY, "cache-hit", "reused paid answer"))
                    return self._finish(
                        prompt, cached, Rung.CLASSIFY, 0.95, cls, trace, started,
                        verified=True, cached=True,
                    )
                trace.append(RungTrace(Rung.CLASSIFY, "cache-rejected", check.reason))

        try:
            return self._climb(prompt, cls, trace, started, time_cap)
        except BudgetExceeded as exc:
            trace.append(RungTrace(Rung.REMOTE_STRONG, "budget-exhausted", str(exc)))
            return self._best_effort(prompt, cls, trace, started)

    # ------------------------------------------------------------------ rungs

    def _climb(
        self,
        prompt: str,
        cls: Classification,
        trace: list[RungTrace],
        started: float,
        time_cap: float,
    ) -> TaskResult:
        candidates: list[_Candidate] = []
        # Normalized answers the remote judge has said NO to: self-agreement
        # by the same small model must never override that verdict.
        rejected: set[str] = set()
        threshold = self._thresholds.get(cls.task_type)

        # Rung 0.5: explicit arithmetic is solved exactly by Python - the one
        # answer source that is both free AND certain.
        if cls.task_type == TaskType.MATH:
            solved = try_solve_math(prompt)
            if solved is not None:
                trace.append(RungTrace(Rung.CLASSIFY, "tool-solved", solved))
                return self._finish(
                    prompt, solved, Rung.CLASSIFY, 0.99, cls, trace, started,
                    verified=True,
                )

        try:
            too_little_time = (
                time_cap < _MIN_LOCAL_ATTEMPT_SECONDS and self._remote is not None
            )
            skip_local = (
                self._local is None
                or cls.difficulty >= self._cfg.skip_ahead_difficulty
                or too_little_time
            )
            if skip_local:
                if self._local is None:
                    reason = "no local model"
                elif too_little_time:
                    reason = f"time cap {time_cap:.0f}s below local minimum"
                else:
                    reason = "difficulty skip-ahead"
                trace.append(RungTrace(Rung.CLASSIFY, "skip-local", reason))
            else:
                result = self._try_local_rungs(
                    prompt, cls, threshold, candidates, trace, started, time_cap,
                    rejected,
                )
                if result is not None:
                    return result

            return self._remote_rungs(prompt, cls, candidates, trace, started, time_cap)
        except BudgetExceeded as exc:
            # Out of paid budget: ship the best free answer we already have.
            trace.append(RungTrace(Rung.REMOTE_STRONG, "budget-exhausted", str(exc)))
            return self._settle_for_best(prompt, cls, candidates, trace, started)

    def _try_local_rungs(
        self,
        prompt: str,
        cls: Classification,
        threshold: float,
        candidates: list[_Candidate],
        trace: list[RungTrace],
        started: float,
        time_cap: float,
        rejected: set[str],
    ) -> TaskResult | None:
        local_prompt = self._local_prompt(prompt, cls.task_type)

        # Rung 1: first local attempt.
        candidate = self._local_attempt(local_prompt, cls, trace, Rung.LOCAL_FIRST)
        if candidate is not None:
            candidates.append(candidate)
            if candidate.verified and candidate.confidence >= threshold:
                result = self._ship_or_judge(
                    prompt, candidate, Rung.LOCAL_FIRST, cls, trace, started, rejected
                )
                if result is not None:
                    return result
        if self._out_of_time(started, time_cap):
            return self._time_pressure_exit(
                prompt, cls, candidates, trace, started, rejected
            )

        # Rung 2: reworded, hotter retry.
        candidate = self._local_attempt(
            local_prompt + _RETRY_SUFFIX, cls, trace, Rung.LOCAL_RETRY,
            temperature=self._local_cfg.retry_temperature,
        )
        if candidate is not None:
            candidates.append(candidate)
            if candidate.verified and candidate.confidence >= threshold:
                result = self._ship_or_judge(
                    prompt, candidate, Rung.LOCAL_RETRY, cls, trace, started, rejected
                )
                if result is not None:
                    return result
        if self._out_of_time(started, time_cap):
            return self._time_pressure_exit(
                prompt, cls, candidates, trace, started, rejected
            )

        # Rung 3: self-consistency - sample up to k answers, majority vote.
        # Early exit: a unanimous quorum needs no more evidence; any dissent
        # disables the shortcut and the full k-sample vote decides.
        while len(candidates) < self._cfg.self_consistency_k:
            winner = self._unanimous_quorum(candidates, rejected)
            if winner is not None:
                trace.append(
                    RungTrace(
                        Rung.SELF_CONSISTENCY, "early-consensus",
                        f"unanimous after {len(candidates)} samples",
                    )
                )
                result = self._ship_unanimous(
                    prompt, winner, 1.0, cls, trace, started, rejected
                )
                if result is not None:
                    return result
                # Judge vetoed the consensus: keep sampling for a new answer.
            if self._out_of_time(started, time_cap):
                break
            candidate = self._local_attempt(
                local_prompt, cls, trace, Rung.SELF_CONSISTENCY,
                temperature=self._local_cfg.retry_temperature,
            )
            if candidate is None:
                break
            candidates.append(candidate)

        verified_texts = [c.text for c in candidates if c.verified]
        if verified_texts:
            winner, ratio = majority_vote(cls.task_type, verified_texts)
            trace.append(
                RungTrace(
                    Rung.SELF_CONSISTENCY, "vote",
                    f"ratio={ratio:.2f} over {len(verified_texts)} verified",
                )
            )
            winner_norm = normalize(cls.task_type, winner)
            if winner_norm in rejected:
                # The judge already vetoed this answer; go straight to remote.
                trace.append(
                    RungTrace(Rung.SELF_CONSISTENCY, "winner-rejected", "judge said NO")
                )
                return None
            vote_confidence = max(ratio, max(c.confidence for c in candidates if c.verified))
            if ratio >= self._cfg.unanimous_ratio:
                result = self._ship_unanimous(
                    prompt, winner, vote_confidence, cls, trace, started, rejected
                )
                if result is not None:
                    return result
                return None  # judge vetoed unanimity: go remote
            # Split-vote tiebreak: a contested-but-leading vote goes to the
            # 1-token remote judge instead of full remote generation.
            if ratio >= self._cfg.contested_ratio and self._cfg.judge_enabled:
                result = self._judge_rung(
                    prompt, winner, vote_confidence, cls, trace, started, rejected
                )
                if result is not None:
                    return result
        return None

    def _time_pressure_exit(
        self,
        prompt: str,
        cls: Classification,
        candidates: list[_Candidate],
        trace: list[RungTrace],
        started: float,
        rejected: set[str],
    ) -> TaskResult | None:
        """Out of local time. Normal types settle for the best free answer;
        weak-verifier types get a fast judge verdict on the best candidate
        (ship on YES) and otherwise return None so the caller escalates to
        remote - a wrong-but-unjudged answer risks the whole accuracy gate.
        """
        if (
            cls.task_type in _JUDGE_REQUIRED_TYPES
            and self._remote is not None
        ):
            best = self._best_candidate(candidates)
            if best is not None and best.verified and self._cfg.judge_enabled:
                result = self._judge_rung(
                    prompt, best.text, best.confidence, cls, trace, started, rejected
                )
                if result is not None:
                    return result
            trace.append(
                RungTrace(Rung.SELF_CONSISTENCY, "time-pressure-escalate", "")
            )
            return None
        return self._settle_for_best(prompt, cls, candidates, trace, started)

    def _ship_unanimous(
        self,
        prompt: str,
        winner: str,
        confidence: float,
        cls: Classification,
        trace: list[RungTrace],
        started: float,
        rejected: set[str],
    ) -> TaskResult | None:
        """Ship a unanimous self-consistency winner, judge-gated for the
        weak-verifier types. None means the judge said NO."""
        needs_judge = (
            cls.task_type in _JUDGE_REQUIRED_TYPES
            and self._remote is not None
            and self._cfg.judge_enabled
        )
        if needs_judge:
            return self._judge_rung(
                prompt, winner, confidence, cls, trace, started, rejected
            )
        self._thresholds.update(cls.task_type, True)
        return self._finish(
            prompt, winner, Rung.SELF_CONSISTENCY, confidence,
            cls, trace, started, verified=True,
        )

    def _ship_or_judge(
        self,
        prompt: str,
        candidate: _Candidate,
        rung: Rung,
        cls: Classification,
        trace: list[RungTrace],
        started: float,
        rejected: set[str],
    ) -> TaskResult | None:
        """Ship a confident rung-1/2 answer, via the judge for weak-verifier
        types. Returns None when the judge says NO (keep climbing)."""
        needs_judge = (
            cls.task_type in _JUDGE_REQUIRED_TYPES
            and self._remote is not None
            and self._cfg.judge_enabled
        )
        if needs_judge:
            if candidate.normalized in rejected:
                return None  # already vetoed; don't pay to re-judge it
            return self._judge_rung(
                prompt, candidate.text, candidate.confidence, cls, trace, started,
                rejected,
            )
        self._thresholds.update(cls.task_type, True)
        return self._finish(
            prompt, candidate.text, rung, candidate.confidence,
            cls, trace, started, verified=True,
        )

    def _judge_rung(
        self,
        prompt: str,
        winner: str,
        vote_confidence: float,
        cls: Classification,
        trace: list[RungTrace],
        started: float,
        rejected: set[str],
    ) -> TaskResult | None:
        """Rung 4: remote verifies the local winner for ~1 output token."""
        if self._remote is None:
            return None
        self._budget.check_remaining(_JUDGE_ESTIMATED_TOKENS)
        try:
            verdict, judge_result = self._remote.judge(
                compress_prompt(prompt), winner
            )
        except GenerationError as exc:
            trace.append(RungTrace(Rung.REMOTE_JUDGE, "judge-error", str(exc)))
            return None
        self._budget.record(judge_result)
        trace.append(
            RungTrace(
                Rung.REMOTE_JUDGE,
                "judge-verdict",
                "YES" if verdict else "NO",
                remote_tokens=judge_result.total_tokens,
            )
        )
        if verdict:
            self._thresholds.update(cls.task_type, True)
            return self._finish(
                prompt, winner, Rung.REMOTE_JUDGE, max(vote_confidence, 0.9),
                cls, trace, started, verified=True,
            )
        self._thresholds.update(cls.task_type, False)
        rejected.add(normalize(cls.task_type, winner))
        return None

    def _remote_rungs(
        self,
        prompt: str,
        cls: Classification,
        candidates: list[_Candidate],
        trace: list[RungTrace],
        started: float,
        time_cap: float,
    ) -> TaskResult:
        if self._remote is None:
            trace.append(
                RungTrace(Rung.REMOTE_CHEAP, "no-remote", "running without a remote client")
            )
            return self._settle_for_best(prompt, cls, candidates, trace, started)

        if candidates:
            # Reaching a paid generation rung means local failed this task type.
            self._thresholds.update(cls.task_type, False)

        compressed = compress_prompt(prompt)
        remote_prompt = self._with_draft_hint(compressed, candidates)

        # Rung 5: cheap remote model, tight output cap.
        self._budget.check_remaining(self._remote_cfg.max_tokens_cheap)
        try:
            result = self._remote.generate(
                remote_prompt,
                max_tokens=self._remote_cfg.max_tokens_cheap,
                model=self._remote_cfg.cheap_model,
            )
            self._budget.record(result)
            trace.append(
                RungTrace(
                    Rung.REMOTE_CHEAP, "remote-cheap",
                    result.model_id, remote_tokens=result.total_tokens,
                )
            )
            check = verify(cls.task_type, prompt, result.text)
            if check.ok:
                self._store_in_cache(prompt, result.text)
                return self._finish(
                    prompt, result.text, Rung.REMOTE_CHEAP, 0.85,
                    cls, trace, started, verified=True,
                )
        except GenerationError as exc:
            trace.append(RungTrace(Rung.REMOTE_CHEAP, "remote-cheap-error", str(exc)))

        if self._out_of_time(started, time_cap):
            # Cheap remote already failed/was unverified; settle rather than
            # spend more time on the strong model.
            return self._settle_for_best(prompt, cls, candidates, trace, started)

        # Rung 6: strong remote model - the last resort.
        self._budget.check_remaining(self._remote_cfg.max_tokens_strong)
        try:
            result = self._remote.generate(
                remote_prompt,
                max_tokens=self._remote_cfg.max_tokens_strong,
                model=self._remote_cfg.strong_model,
            )
            self._budget.record(result)
            trace.append(
                RungTrace(
                    Rung.REMOTE_STRONG, "remote-strong",
                    result.model_id, remote_tokens=result.total_tokens,
                )
            )
            check = verify(cls.task_type, prompt, result.text)
            self._store_in_cache(prompt, result.text)
            return self._finish(
                prompt, result.text, Rung.REMOTE_STRONG, 0.9,
                cls, trace, started, verified=check.ok,
            )
        except GenerationError as exc:
            trace.append(RungTrace(Rung.REMOTE_STRONG, "remote-strong-error", str(exc)))
            return self._settle_for_best(prompt, cls, candidates, trace, started)

    # ---------------------------------------------------------------- helpers

    def _local_attempt(
        self,
        local_prompt: str,
        cls: Classification,
        trace: list[RungTrace],
        rung: Rung,
        temperature: float | None = None,
    ) -> _Candidate | None:
        try:
            result: GenerationResult = self._local.generate(
                local_prompt,
                temperature=temperature,
                max_tokens=self._local_cfg.max_tokens_by_type.get(
                    str(cls.task_type), self._local_cfg.max_tokens
                ),
            )
        except GenerationError as exc:
            trace.append(RungTrace(rung, "local-error", str(exc)))
            return None
        self._budget.record(result)
        check = verify(cls.task_type, local_prompt, result.text)
        confidence = logprob_to_confidence(result.logprob_mean)
        trace.append(
            RungTrace(
                rung, "local-attempt",
                f"verified={check.ok} confidence={confidence:.2f}"
                + (f" reason={check.reason}" if not check.ok else ""),
            )
        )
        return _Candidate(result.text, confidence, check.ok, check.normalized)

    def _local_prompt(self, prompt: str, task_type: TaskType) -> str:
        return f"{prompt}\n\n{_LOCAL_INSTRUCTIONS[task_type]}"

    def _with_draft_hint(self, compressed: str, candidates: list[_Candidate]) -> str:
        """Local-CoT -> remote-answer: ship the local draft when it is short."""
        best = self._best_candidate(candidates)
        if best is None or not best.normalized:
            return compressed
        if len(best.normalized) > _DRAFT_HINT_MAX_CHARS:
            return compressed
        return (
            f"{compressed}\n\n"
            f"A draft answer (may be wrong): {best.normalized}\n"
            "If the draft is correct, repeat it; otherwise give the correct answer."
        )

    def _unanimous_quorum(
        self, candidates: list[_Candidate], rejected: set[str]
    ) -> str | None:
        """Winner text when >= quorum verified answers agree with no dissent.

        A judge-rejected answer never wins: the same model repeating itself
        is not evidence against an explicit remote NO.
        """
        verified = [c for c in candidates if c.verified and c.normalized]
        if len(verified) < self._cfg.early_consensus_quorum:
            return None
        distinct = {c.normalized for c in verified}
        if len(distinct) != 1 or verified[0].normalized in rejected:
            return None
        return verified[0].text

    @staticmethod
    def _best_candidate(candidates: list[_Candidate]) -> _Candidate | None:
        if not candidates:
            return None
        verified = [c for c in candidates if c.verified]
        pool = verified or candidates
        return max(pool, key=lambda c: c.confidence)

    def _settle_for_best(
        self,
        prompt: str,
        cls: Classification,
        candidates: list[_Candidate],
        trace: list[RungTrace],
        started: float,
    ) -> TaskResult:
        """Caps hit: ship the best free answer we have rather than nothing."""
        best = self._best_candidate(candidates)
        trace.append(RungTrace(Rung.SELF_CONSISTENCY, "settled", "caps reached"))
        if best is None:
            return self._finish(
                prompt, "", Rung.SELF_CONSISTENCY, 0.0, cls, trace, started, verified=False
            )
        return self._finish(
            prompt, best.text, Rung.SELF_CONSISTENCY, best.confidence,
            cls, trace, started, verified=best.verified,
        )

    def _best_effort(
        self,
        prompt: str,
        cls: Classification,
        trace: list[RungTrace],
        started: float,
    ) -> TaskResult:
        return self._settle_for_best(prompt, cls, [], trace, started)

    def _store_in_cache(self, prompt: str, answer: str) -> None:
        if self._cache is not None:
            self._cache.put(prompt, answer)

    def _out_of_time(self, started: float, time_cap: float) -> bool:
        return (time.monotonic() - started) > time_cap

    def _finish(
        self,
        prompt: str,
        answer: str,
        rung: Rung,
        confidence: float,
        cls: Classification,
        trace: list[RungTrace],
        started: float,
        *,
        verified: bool,
        cached: bool = False,
    ) -> TaskResult:
        remote_tokens = self._budget.end_task(rung)
        return TaskResult(
            answer=answer,
            exit_rung=rung,
            confidence=confidence,
            remote_tokens=remote_tokens,
            task_type=cls.task_type,
            cached=cached,
            verified=verified,
            elapsed_seconds=time.monotonic() - started,
            trace=tuple(trace),
        )
