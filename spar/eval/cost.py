"""Token/cost accounting (design §5.6). Cost is float USD reporting metadata — Decimal stays
reserved for simulator money. `litellm` is never imported here (provider cost arrives already
parsed on CallUsage.response_cost). Two numbers: pre-flight `estimate_cost` and confirmed
per-call accrual via `CostMeter` (which enforces the hard budget cap)."""

from __future__ import annotations

import threading
from collections.abc import Callable

from spar.dataset.loader import load_split
from spar.eval.agent import CallUsage
from spar.eval.models import ModelConfig
from spar.eval.profile import Profile
from spar.simulator.schemas import Sample


class BudgetExceeded(RuntimeError):
    """Raised by _cached_completion_fn before a paid completion when the cost meter is over budget.
    Must NOT be caught as an infra error (it halts the run, not retried)."""


def _confirmed_cost(usage: CallUsage, model: ModelConfig) -> float:
    """Prefer the provider-reported cost; else price tokens off models.toml; else 0.0."""
    if usage.response_cost is not None:
        return float(usage.response_cost)
    pin = model.price_in_per_mtok
    pout = model.price_out_per_mtok
    if pin is None or pout is None:
        return 0.0
    return (usage.prompt_tokens / 1_000_000) * pin + (usage.completion_tokens / 1_000_000) * pout


class CostMeter:
    """Accrues agent inference cost and enforces the hard budget cap (design §5.6). Tracks TWO
    numbers (a cache HIT is `billed=False`):

    - `gross()`  — what the run WOULD cost with no cache: every completion (hits + misses). This is
      the true compute cost of the work, independent of resume state.
    - `spent()` / `billed()` — what was ACTUALLY spent this run: cache MISSES only (real paid calls).
      The budget cap keys on THIS, so a fully-cached resume is free and never trips the cap.

    On a fresh run (empty cache) gross == spent; on a resume spent < gross. Only ModelAgent.usage is
    recorded here; the pinned judge/user-sim are metered separately as overhead (see record_overhead)."""

    def __init__(self, *, budget_usd: float | None) -> None:
        self._budget = budget_usd
        self._billed = 0.0          # actually spent this run (cache misses) — drives the cap
        self._gross = 0.0           # uncached would-be cost (hits + misses)
        self._overhead = 0.0        # judge/user-sim spend (real money, also capped)
        # Guards every read/write so the meter stays exact under concurrent sample workers
        # (Tier-2b). Without it, two threads doing `_billed += cost` can lose an update and
        # under-count spend, silently weakening the budget cap.
        self._lock = threading.Lock()

    def record(self, usage: CallUsage, model: ModelConfig) -> None:
        # Round each running total to sub-cent precision so accumulated float USD stays clean
        # (avoids IEEE-754 drift like 0.40 + 0.20 == 0.6000000000000001).
        cost = _confirmed_cost(usage, model)
        with self._lock:
            self._gross = round(self._gross + cost, 10)
            if usage.billed:
                self._billed = round(self._billed + cost, 10)

    def set_overhead(self, total_usd: float) -> None:
        """Set the cumulative judge/user-sim (responder/grader) spend so far — real money that is
        billed and counts toward the cap, but reported separately from the agent's cost_usd. The
        responder/grader track their own running total; the orchestrator syncs it here per sample."""
        with self._lock:
            self._overhead = round(total_usd, 10)

    def spent(self) -> float:
        """Actual agent money spent this run (cache misses only) — the published cost_usd."""
        with self._lock:
            return self._billed

    def billed(self) -> float:
        """Alias of spent(): agent money actually spent this run."""
        return self.spent()

    def gross(self) -> float:
        """What the run would cost with no cache (every completion, hits included)."""
        with self._lock:
            return self._gross

    def overhead(self) -> float:
        """Judge/user-sim (responder/grader) money spent this run."""
        with self._lock:
            return self._overhead

    def over_budget(self) -> bool:
        # Cap on ACTUAL money out the door: agent (billed) + responder/grader overhead. Cache
        # replays (gross-only) never count, so a resume stays free even when over budget.
        with self._lock:
            return self._budget is not None and (self._billed + self._overhead) >= self._budget


# Heuristic per-request token sizes for the pre-flight estimate (deliberately coarse, §5.6):
# a system prompt (policy + tool catalog) plus a redacted observation per turn, and a short
# JSON action out. Real per-episode turn count + output length are emergent; this only informs
# the launch decision and the cap.
_EST_PROMPT_TOKENS_PER_TURN = 1500
_EST_COMPLETION_TOKENS_PER_TURN = 200


def _split_size(split: str) -> int:
    try:
        return len(load_split(split))
    except Exception:
        # An unconfigured release falls back to the bundled toy split; never fail a dry-run.
        return len(load_split("lite"))


def estimate_cost(
    models: list[ModelConfig], profile: Profile, *, avg_turns: int = 6,
    samples_for: Callable[[str], list[Sample]] | None = None,
) -> dict[str, float]:
    """PRE-FLIGHT heuristic projection (no model calls). For each StagePlan:
    n_samples x k x avg_turns requests, each a fixed prompt/completion token budget, priced off
    models.toml. Unpriced models project to 0.0. `samples_for` (when given) sizes against the real
    release; default sizes the bundled toy split. Approximate by design (§5.6)."""
    size = (lambda s: len(samples_for(s))) if samples_for is not None else _split_size
    est: dict[str, float] = {}
    for model in models:
        pin = model.price_in_per_mtok
        pout = model.price_out_per_mtok
        if pin is None or pout is None:
            est[model.id] = 0.0
            continue
        total = 0.0
        for plan in profile.plan:
            requests = size(plan.split) * plan.k * avg_turns
            prompt_mtok = requests * _EST_PROMPT_TOKENS_PER_TURN / 1_000_000
            completion_mtok = requests * _EST_COMPLETION_TOKENS_PER_TURN / 1_000_000
            total += prompt_mtok * pin + completion_mtok * pout
        est[model.id] = total
    return est
