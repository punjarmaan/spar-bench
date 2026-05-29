"""Batch orchestrator (design §5.5): drive split x k with the typed failure taxonomy,
infra retries (exp backoff + jitter), cache/resume, confirmed cost + hard cap, per-episode
audit trajectories, and a reproducibility run_manifest. Writes runs/<id>/<split>.results.json
via the unchanged build_results. litellm is never imported at module load (lazy in the agent)."""

from __future__ import annotations

import random
from collections.abc import Callable
from enum import Enum
from typing import Any, TypeVar

from spar.eval.agent import agent_factory
from spar.eval.cache import CompletionCache, cache_key
from spar.eval.cost import CostMeter
from spar.eval.models import ModelConfig
from spar.eval.profile import StageSampling
from spar.harness.graders import ModelGrader, SampleScore, score
from spar.harness.passk import is_solved
from spar.harness.runner import run_episode
from spar.harness.user_sim import UserSim
from spar.policies.loader import load_policy
from spar.simulator.contract import Abort
from spar.simulator.enums import Axis

_T = TypeVar("_T")

PUBLISHABILITY_FLOOR = 0.98   # design §5.5: scored_fraction >= floor -> "verified", else "partial"


class SampleStatus(str, Enum):
    SCORED = "scored"                  # reached a terminal and graded normally; in denominator
    ERRORED_INFRA = "errored_infra"    # timeout/429/5xx/context-overflow after retries; excluded
    MALFORMED_ACTION = "malformed_action"  # no valid JSON after reformat retry; scored as Abort
    REFUSED = "refused"                # model refused in-band; scored as its Abort/escalate


# Substrings that mark a failure as OURS (infra), not the model's behaviour (design §5.5/§8).
_INFRA_SIGNATURES = (
    "429", "rate limit", "rate_limit", "too many requests",
    "500", "502", "503", "504", "service unavailable", "bad gateway",
    "timeout", "timed out",
    "context_length_exceeded", "context length", "maximum context", "context-overflow",
)


def _is_infra_error(exc: BaseException) -> bool:
    """True for timeout / 429 / 5xx / context-overflow (retryable, never a capability fail)."""
    if isinstance(exc, TimeoutError):
        return True
    text = f"{type(exc).__name__} {exc}".lower()
    return any(sig in text for sig in _INFRA_SIGNATURES)


def _with_retries(
    call: Callable[[], _T],
    *,
    retries: int,
    sleep: Callable[[float], None],
    jitter: Callable[[], float] = random.random,
    base_delay: float = 0.5,
) -> _T:
    """Run `call`, retrying ONLY infra errors with exponential backoff + jitter up to `retries`
    extra attempts. `sleep`/`jitter` are injected so tests run instantly. Non-infra exceptions
    propagate on the first occurrence. Re-raises the last infra error once retries are exhausted."""
    attempt = 0
    while True:
        try:
            return call()
        except BaseException as exc:  # noqa: BLE001 - re-raised below unless retryable
            if not _is_infra_error(exc) or attempt >= retries:
                raise
            delay = base_delay * (2 ** attempt) + jitter()
            sleep(delay)
            attempt += 1


class _CachedMsg:
    def __init__(self, content: str) -> None:
        self.content = content


class _CachedChoice:
    def __init__(self, content: str) -> None:
        self.message = _CachedMsg(content)


class _CachedResponse:
    """Replays a cached completion in the litellm response shape the agent reads."""

    def __init__(self, payload: dict) -> None:
        self.choices = [_CachedChoice(payload["content"])]
        self.usage = type("U", (), {
            "prompt_tokens": payload.get("prompt_tokens", 0),
            "completion_tokens": payload.get("completion_tokens", 0),
        })()
        self._hidden_params = {"response_cost": payload.get("response_cost")}


def _extract(resp: Any) -> dict:
    """Pull the cache-storable fields out of any litellm-shaped response object."""
    usage = getattr(resp, "usage", None)
    hidden = getattr(resp, "_hidden_params", {}) or {}
    return {
        "content": resp.choices[0].message.content,
        "prompt_tokens": getattr(usage, "prompt_tokens", 0),
        "completion_tokens": getattr(usage, "completion_tokens", 0),
        "response_cost": hidden.get("response_cost"),
    }


def _cached_completion_fn(
    raw_fn: Callable[..., Any],
    cache: CompletionCache,
    *,
    retries: int,
    sleep: Callable[[float], None],
) -> Callable[..., _CachedResponse]:
    """Wrap a completion_fn with the content-addressed cache + infra retries. A cache hit replays
    the stored response and makes NO underlying call (resume idempotency, design §5.5/§5.6)."""

    def fn(*, model: str, messages: list[dict], **sampling: Any) -> _CachedResponse:
        key = cache_key(model, messages, sampling)
        cached = cache.get(key)
        if cached is not None:
            return _CachedResponse(cached)
        resp = _with_retries(
            lambda: raw_fn(model=model, messages=messages, **sampling),
            retries=retries, sleep=sleep,
        )
        payload = _extract(resp)
        cache.put(key, payload)
        return _CachedResponse(payload)

    return fn


def _classify(score_obj: SampleScore, trace) -> SampleStatus:
    """Tag the model-behaviour sub-case of a graded sample (design §5.5).

    A graded sample is SCORED unless the agent layer flagged a malformed action or an in-band
    refusal via the terminating Abort it emitted (EM1 sets reason="malformed_action")."""
    last = trace.action_log[-1] if trace.action_log else None
    if isinstance(last, Abort):
        if last.reason == "malformed_action":
            return SampleStatus.MALFORMED_ACTION
        if trace.made_decision and trace.abort_reason != "step_budget_exhausted":
            return SampleStatus.REFUSED
    return SampleStatus.SCORED


def _run_sample(
    sample,
    model: ModelConfig,
    *,
    sampling: StageSampling,
    completion_fn: Callable[..., Any],
    cache: CompletionCache,
    meter: CostMeter,
    responder: UserSim,
    grader: ModelGrader,
    k: int,
    retries: int,
    sleep: Callable[[float], None],
) -> tuple[SampleStatus, SampleScore | None]:
    """Run one sample's pass^k offline. Infra errors that survive retries -> ERRORED_INFRA (no
    score, excluded from the denominator but counted). Otherwise grade + classify the status and
    accrue confirmed agent cost."""
    cached_fn = _cached_completion_fn(completion_fn, cache, retries=retries, sleep=sleep)
    policy_text = load_policy(sample.policy_id)
    factory = agent_factory(
        model,
        policy_text=policy_text,
        sampling=sampling,
        completion_fn=cached_fn,
        mandate_text=sample.mandate.goal,
    )
    binary = sample.axis is not Axis.ROUTING
    solved = 0
    last_trace = None
    last_agent = None
    for trial_index in range(k):
        agent = factory()
        try:
            trace = run_episode(sample, agent, trial_index=trial_index, user_sim=responder)
        except BaseException as exc:  # noqa: BLE001
            if _is_infra_error(exc):
                # Cost of any calls that DID return before the fatal infra error is still real.
                for u in getattr(agent, "usage", []):
                    meter.record(u, model)
                return SampleStatus.ERRORED_INFRA, None
            raise
        for u in agent.usage:
            meter.record(u, model)
        if is_solved(score(sample, trace, model_grader=grader).score, binary=binary):
            solved += 1
        last_trace = trace
        last_agent = agent
    assert last_trace is not None and last_agent is not None
    final = score(sample, last_trace, model_grader=grader)
    final.trials_n = k
    final.trials_c = solved
    return _classify(final, last_trace), final
