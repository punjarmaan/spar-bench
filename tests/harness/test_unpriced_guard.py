"""Unpriced-model budget guard: an unpriced model meters at $0 if the provider omits
response_cost, silently defeating the --budget-usd cap. The eval CLI warns before any spend.
"""

from __future__ import annotations

from spar.eval.models import ModelConfig
from spar.harness.run_eval import _unpriced_warnings


def _m(id: str, *, priced: bool) -> ModelConfig:
    kw = dict(id=id, route=f"openrouter/x/{id}", cls="open")
    if priced:
        kw["price_in_per_mtok"] = 1.0
        kw["price_out_per_mtok"] = 2.0
    return ModelConfig(**kw)  # type: ignore[arg-type]


def test_warns_for_unpriced_model_when_budget_set() -> None:
    models = [_m("priced", priced=True), _m("free", priced=False)]
    warns = _unpriced_warnings(models, only=None, budget_usd=2.0)
    assert len(warns) == 1
    assert "free" in warns[0]
    assert "cap will NOT fire" in warns[0]


def test_no_warning_without_a_budget() -> None:
    models = [_m("free", priced=False)]
    assert _unpriced_warnings(models, only=None, budget_usd=None) == []


def test_only_filter_scopes_the_warning() -> None:
    models = [_m("priced", priced=True), _m("free", priced=False)]
    # Running only the priced model -> no warning even though an unpriced model is in the roster.
    assert _unpriced_warnings(models, only="priced", budget_usd=2.0) == []
    assert len(_unpriced_warnings(models, only="free", budget_usd=2.0)) == 1


def test_priced_roster_is_silent() -> None:
    models = [_m("a", priced=True), _m("b", priced=True)]
    assert _unpriced_warnings(models, only=None, budget_usd=5.0) == []
