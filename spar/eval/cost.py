"""Token/cost accounting (design §5.6). Cost is float USD reporting metadata — Decimal stays
reserved for simulator money. `litellm` is never imported here (provider cost arrives already
parsed on CallUsage.response_cost). Two numbers: pre-flight `estimate_cost` and confirmed
per-call accrual via `CostMeter` (which enforces the hard budget cap)."""

from __future__ import annotations

from spar.eval.agent import CallUsage
from spar.eval.models import ModelConfig
from spar.eval.profile import Profile


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
    """Accrues CONFIRMED agent+judge+user-sim spend and enforces the cap (design §5.6)."""

    def __init__(self, *, budget_usd: float | None) -> None:
        self._budget = budget_usd
        self._spent = 0.0

    def record(self, usage: CallUsage, model: ModelConfig) -> None:
        # Round the running total to sub-cent precision so accumulated float USD stays clean
        # (avoids IEEE-754 drift like 0.40 + 0.20 == 0.6000000000000001).
        self._spent = round(self._spent + _confirmed_cost(usage, model), 10)

    def spent(self) -> float:
        return self._spent

    def over_budget(self) -> bool:
        return self._budget is not None and self._spent >= self._budget


def estimate_cost(
    models: list[ModelConfig], profile: Profile, *, avg_turns: int = 6
) -> dict[str, float]:
    """PRE-FLIGHT heuristic projection (no model calls) — design §5.6. Filled in Task 4."""
    raise NotImplementedError
