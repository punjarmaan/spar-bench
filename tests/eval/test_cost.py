"""EM2 Tasks 3-4: confirmed-cost accrual, budget cap, and the pre-flight estimate."""

from __future__ import annotations

from spar.eval.cost import CostMeter, estimate_cost
from spar.eval.models import ModelConfig
from spar.eval.agent import CallUsage
from spar.eval.profile import DEFAULT_PROFILE


def _model(**over) -> ModelConfig:
    base = dict(
        id="m", route="openrouter/x/y", cls="open",
        price_in_per_mtok=10.0, price_out_per_mtok=30.0,
    )
    base.update(over)
    return ModelConfig(**base)


def test_record_prefers_provider_response_cost() -> None:
    meter = CostMeter(budget_usd=None)
    meter.record(CallUsage(prompt_tokens=1000, completion_tokens=500, response_cost=0.25), _model())
    assert meter.spent() == 0.25


def test_record_falls_back_to_token_pricing() -> None:
    meter = CostMeter(budget_usd=None)
    # 1M in @ $10 + 1M out @ $30 = $40.0
    meter.record(
        CallUsage(prompt_tokens=1_000_000, completion_tokens=1_000_000, response_cost=None),
        _model(),
    )
    assert abs(meter.spent() - 40.0) < 1e-9


def test_record_zero_when_no_cost_and_no_prices() -> None:
    meter = CostMeter(budget_usd=None)
    meter.record(
        CallUsage(prompt_tokens=100, completion_tokens=50, response_cost=None),
        _model(price_in_per_mtok=None, price_out_per_mtok=None),
    )
    assert meter.spent() == 0.0


def test_over_budget_trips_at_cap() -> None:
    meter = CostMeter(budget_usd=0.50)
    assert not meter.over_budget()
    meter.record(CallUsage(prompt_tokens=0, completion_tokens=0, response_cost=0.40), _model())
    assert not meter.over_budget()
    meter.record(CallUsage(prompt_tokens=0, completion_tokens=0, response_cost=0.20), _model())
    assert meter.spent() == 0.60
    assert meter.over_budget()


def test_no_budget_never_over() -> None:
    meter = CostMeter(budget_usd=None)
    meter.record(CallUsage(prompt_tokens=0, completion_tokens=0, response_cost=1000.0), _model())
    assert not meter.over_budget()


def test_estimate_cost_positive_and_scales_with_price() -> None:
    cheap = _model(id="cheap", price_in_per_mtok=1.0, price_out_per_mtok=2.0)
    dear = _model(id="dear", price_in_per_mtok=100.0, price_out_per_mtok=200.0)
    est = estimate_cost([cheap, dear], DEFAULT_PROFILE, avg_turns=6)
    assert set(est) == {"cheap", "dear"}
    assert est["cheap"] > 0.0
    assert est["dear"] > est["cheap"]


def test_estimate_cost_unpriced_model_is_zero() -> None:
    free = _model(id="free", price_in_per_mtok=None, price_out_per_mtok=None)
    est = estimate_cost([free], DEFAULT_PROFILE, avg_turns=6)
    assert est["free"] == 0.0


def test_estimate_cost_makes_no_model_calls() -> None:
    # purely arithmetic over loaded sample counts — must not need a completion_fn
    est = estimate_cost([_model()], DEFAULT_PROFILE)
    assert isinstance(est["m"], float)
