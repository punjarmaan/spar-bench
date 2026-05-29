"""Batch orchestrator (design §5.5): drive split x k with the typed failure taxonomy,
infra retries (exp backoff + jitter), cache/resume, confirmed cost + hard cap, per-episode
audit trajectories, and a reproducibility run_manifest. Writes runs/<id>/<split>.results.json
via the unchanged build_results. litellm is never imported at module load (lazy in the agent)."""

from __future__ import annotations

import random
from collections.abc import Callable
from enum import Enum
from typing import TypeVar

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
