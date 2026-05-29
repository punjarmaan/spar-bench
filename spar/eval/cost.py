"""Token/cost accounting (design §5.6). Cost is float USD reporting metadata — Decimal stays
reserved for simulator money. `litellm` is never imported here (provider cost arrives already
parsed on CallUsage.response_cost). Two numbers: pre-flight `estimate_cost` and confirmed
per-call accrual via `CostMeter` (which enforces the hard budget cap)."""

from __future__ import annotations

from spar.dataset.loader import load_split
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
    models: list[ModelConfig], profile: Profile, *, avg_turns: int = 6
) -> dict[str, float]:
    """PRE-FLIGHT heuristic projection (no model calls). For each StagePlan:
    n_samples x k x avg_turns requests, each a fixed prompt/completion token budget, priced off
    models.toml. Unpriced models project to 0.0. Approximate by design (§5.6)."""
    est: dict[str, float] = {}
    for model in models:
        pin = model.price_in_per_mtok
        pout = model.price_out_per_mtok
        if pin is None or pout is None:
            est[model.id] = 0.0
            continue
        total = 0.0
        for plan in profile.plan:
            requests = _split_size(plan.split) * plan.k * avg_turns
            prompt_mtok = requests * _EST_PROMPT_TOKENS_PER_TURN / 1_000_000
            completion_mtok = requests * _EST_COMPLETION_TOKENS_PER_TURN / 1_000_000
            total += prompt_mtok * pin + completion_mtok * pout
        est[model.id] = total
    return est
