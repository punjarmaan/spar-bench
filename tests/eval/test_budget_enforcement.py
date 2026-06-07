"""Per-completion budget cap — BudgetExceeded raised BEFORE raw is called when
over budget; a cache HIT is free and must NOT raise even when over budget."""

from __future__ import annotations

import pytest

from spar.eval.agent import CallUsage
from spar.eval.cache import CompletionCache
from spar.eval.cost import BudgetExceeded, CostMeter
from spar.eval.models import ModelConfig
from spar.eval.orchestrator import _cached_completion_fn


def _model() -> ModelConfig:
    return ModelConfig(
        id="smoke", route="fake/smoke", cls="open",
        version_pin="pin", supports_response_format=False,
    )


def _make_resp() -> object:
    """Minimal litellm-shaped response object: $1.00 per call."""
    class _Msg:
        content = '{"tool":"abort","args":{}}'

    class _Choice:
        message = _Msg()

    class _Usage:
        prompt_tokens = 1000
        completion_tokens = 100

    class _Resp:
        choices = [_Choice()]
        usage = _Usage()
        _hidden_params = {"response_cost": 1.0}

    return _Resp()


def test_cached_fn_raises_before_calling_when_over_budget(tmp_path: object) -> None:
    """raw must be called exactly once; the second call raises BudgetExceeded WITHOUT entering raw."""
    assert isinstance(tmp_path, type(tmp_path))  # tmp_path is pytest's Path fixture
    cache = CompletionCache(tmp_path)  # type: ignore[arg-type]
    meter = CostMeter(budget_usd=0.5)  # cap is $0.50; each call costs $1.00
    calls: dict[str, int] = {"n": 0}

    def raw(*, model: str, messages: object, **kw: object) -> object:
        calls["n"] += 1
        return _make_resp()

    fn = _cached_completion_fn(
        raw, cache, retries=0, sleep=lambda _: None, trial_index=0, meter=meter
    )

    # First call: cache miss, meter is NOT yet over budget ($0 < $0.50), so raw is entered.
    fn(model="m", messages=[{"role": "user", "content": "x"}], temperature=0.0)
    assert calls["n"] == 1, "raw must be called once on the first miss"

    # Drive the meter over budget (mirrors what _run_sample does after run_episode finishes).
    meter.record(CallUsage(prompt_tokens=1000, completion_tokens=100, response_cost=1.0), _model())
    assert meter.over_budget(), "meter must be over budget before the second call"

    # Second call: cache miss (different messages), meter IS over budget -> must raise BEFORE raw.
    with pytest.raises(BudgetExceeded):
        fn(model="m", messages=[{"role": "user", "content": "y"}], temperature=0.0)

    assert calls["n"] == 1, "raw must NOT be entered while over budget"


def test_cache_hit_is_free_when_over_budget(tmp_path: object) -> None:
    """A cache HIT must NEVER raise BudgetExceeded — replaying a cached response makes no paid call.
    This is the key safety invariant: resume after a budget halt must replay for free."""
    cache = CompletionCache(tmp_path)  # type: ignore[arg-type]
    # Budget already exhausted before we even start.
    meter = CostMeter(budget_usd=0.0)
    meter.record(CallUsage(prompt_tokens=0, completion_tokens=0, response_cost=0.01), _model())
    assert meter.over_budget()

    calls: dict[str, int] = {"n": 0}

    def raw(*, model: str, messages: object, **kw: object) -> object:
        calls["n"] += 1
        return _make_resp()

    fn = _cached_completion_fn(
        raw, cache, retries=0, sleep=lambda _: None, trial_index=0, meter=meter
    )

    # Seed the cache manually: we need a hit, so call fn WITHOUT the meter guard.
    # Use a meter with no budget to warm the cache on the first call.
    warm_meter = CostMeter(budget_usd=None)
    fn_warm = _cached_completion_fn(
        raw, cache, retries=0, sleep=lambda _: None, trial_index=0, meter=warm_meter
    )
    msgs = [{"role": "user", "content": "cached"}]
    fn_warm(model="m", messages=msgs, temperature=0.0)
    assert calls["n"] == 1

    # Now replay the SAME messages via the over-budget fn — it must be a cache HIT and NOT raise.
    result = fn(model="m", messages=msgs, temperature=0.0)
    assert result is not None, "cache hit must return a response"
    assert calls["n"] == 1, "raw must NOT be called on a cache hit"
