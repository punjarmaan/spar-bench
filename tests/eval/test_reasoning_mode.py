"""TDD tests for native reasoning mode (feat/reasoning).

All tests use injected fake completion_fns — NO live model calls, NO spend.
"""

from __future__ import annotations

from typing import Any


from spar.eval.models import ModelConfig, load_models
from spar.eval.agent import ModelAgent, agent_factory, REASONING_EFFORT, REASONING_MAX_TOKENS
from spar.eval.profile import StageSampling
from spar.simulator.contract import Abort


# ---------------------------------------------------------------------------
# Minimal fixtures
# ---------------------------------------------------------------------------

_SAMPLING = StageSampling(temperature=0.0, top_p=1.0, max_tokens=512, seed=None)

_POLICY = "You are a payment agent."
_MANDATE = "Complete the payment."


def _make_model(*, reasoning: bool) -> ModelConfig:
    """Minimal ModelConfig with reasoning flag set."""
    return ModelConfig(
        id="test-model",
        route="openrouter/fake/model",
        cls="open",
        supports_response_format=False,
        reasoning=reasoning,
    )


class _RecordingFake:
    """Fake completion_fn that records every set of sampling kwargs it receives.

    Returns a valid capture action on first call so the agent doesn't loop.
    """

    def __init__(self, *, content: str | None = None) -> None:
        self._content = content
        self.recorded_kwargs: list[dict[str, Any]] = []

    def __call__(self, *, model: str, messages: list[dict[str, str]], **sampling: Any) -> Any:
        self.recorded_kwargs.append(dict(sampling))
        # Build a fake litellm-shaped response.
        return _FakeResp(self._content)


class _NullContentFake:
    """Returns content=None (reasoning model with include_reasoning missing or similar)."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, *, model: str, messages: list[dict[str, str]], **sampling: Any) -> Any:
        self.calls += 1
        return _FakeResp(None)


class _Msg:
    def __init__(self, content: str | None) -> None:
        self.content = content


class _Choice:
    def __init__(self, content: str | None) -> None:
        self.message = _Msg(content)


class _Usage:
    prompt_tokens = 10
    completion_tokens = 5


class _FakeResp:
    def __init__(self, content: str | None) -> None:
        self.choices = [_Choice(content)]
        self.usage = _Usage()
        self._hidden_params = {"response_cost": None}
        self.cache_hit = False


def _make_observation():
    """Return a minimal Observation the agent can receive."""
    from decimal import Decimal
    from spar.simulator.contract import Observation, ObsCart, ObsContext, ObsMethod
    from spar.simulator.mandates import IntentMandate

    return Observation(
        mandate=IntentMandate(
            goal="buy a widget", amount_limit=Decimal("100"), currency="USD",
            human_present=True, conditions={},
            per_txn_max=Decimal("200"), daily_remaining=Decimal("350"),
            merchant_constraint=["acme"], mcc_constraint=None,
            allowed_instruments=["visa"], session_ttl_steps=10,
            single_use_or_recurring="single_use", time_window=None,
        ),
        cart=ObsCart(line_items=[{"sku": "w1", "qty": 1}], subtotal=Decimal("90")),
        methods=[ObsMethod(acquirer_id="acq_a", methods=["visa"], geos=["US"],
                           advertised_fee_bps=200, observed_approval_band="high")],
        context=ObsContext(buyer_geo="US", elapsed_steps=0),
    )


# ---------------------------------------------------------------------------
# Part 1 / Part 4 — ModelConfig.reasoning field
# ---------------------------------------------------------------------------

def test_model_config_reasoning_defaults_false():
    """ModelConfig.reasoning should default to False when not specified."""
    cfg = ModelConfig(id="x", route="r", cls="open")
    assert cfg.reasoning is False


def test_model_config_reasoning_true():
    """ModelConfig.reasoning=True should round-trip cleanly."""
    cfg = ModelConfig(id="x", route="r", cls="open", reasoning=True)
    assert cfg.reasoning is True


def test_model_config_reasoning_parses_from_toml(tmp_path):
    """reasoning=true in TOML should parse to True; absent means False."""
    toml = (
        '[[model]]\n'
        'id = "capable"\n'
        'route = "openrouter/anthropic/claude-sonnet-4"\n'
        'class = "frontier"\n'
        'reasoning = true\n'
        '\n'
        '[[model]]\n'
        'id = "plain"\n'
        'route = "openrouter/qwen/qwen3"\n'
        'class = "open"\n'
    )
    path = tmp_path / "m.toml"
    path.write_text(toml, encoding="utf-8")
    models = load_models(path)
    assert models[0].reasoning is True
    assert models[1].reasoning is False


# ---------------------------------------------------------------------------
# Part 3 — _sampling_kwargs for reasoning vs. non-reasoning models
# ---------------------------------------------------------------------------

def test_reasoning_model_receives_reasoning_kwargs():
    """A reasoning=True model should have reasoning_effort, include_reasoning, drop_params in kwargs."""
    fake = _RecordingFake(
        content='{"tool": "abort", "args": {"reason": "malformed_action"}}'
    )
    agent = ModelAgent(
        route="openrouter/fake/model",
        policy_text=_POLICY,
        sampling=_SAMPLING,
        supports_response_format=False,
        completion_fn=fake,
        mandate_text=_MANDATE,
        reasoning=True,
    )
    obs = _make_observation()
    agent.act(obs)

    assert len(fake.recorded_kwargs) >= 1
    first_kwargs = fake.recorded_kwargs[0]
    assert first_kwargs.get("reasoning_effort") == REASONING_EFFORT
    assert first_kwargs.get("include_reasoning") is True  # OpenRouter route
    assert first_kwargs.get("drop_params") is True
    # reasoning overrides the profile max_tokens so thinking can't truncate the JSON answer
    assert first_kwargs.get("max_tokens") == REASONING_MAX_TOKENS


def test_native_reasoning_route_omits_unsupported_sampling():
    """Native Anthropic reasoning routes must omit temperature/top_p and the OpenRouter-only
    include_reasoning (their thinking APIs reject fixed sampling); reasoning_effort, drop_params, and
    the raised max_tokens still apply."""
    for route in ("anthropic/claude-opus-4-8", "anthropic/claude-sonnet-4-6",):
        fake = _RecordingFake(content='{"tool": "abort", "args": {"reason": "malformed_action"}}')
        agent = ModelAgent(
            route=route, policy_text=_POLICY, sampling=_SAMPLING,
            supports_response_format=False, completion_fn=fake, mandate_text=_MANDATE,
            reasoning=True,
        )
        agent.act(_make_observation())
        kw = fake.recorded_kwargs[0]
        assert "temperature" not in kw, route
        assert "top_p" not in kw, route
        assert "include_reasoning" not in kw, route
        assert kw.get("reasoning_effort") == REASONING_EFFORT
        assert kw.get("drop_params") is True
        assert kw.get("max_tokens") == REASONING_MAX_TOKENS


def test_non_reasoning_model_does_not_receive_reasoning_kwargs():
    """A reasoning=False model must NOT include reasoning kwargs — unchanged behavior."""
    fake = _RecordingFake(
        content='{"tool": "abort", "args": {"reason": "malformed_action"}}'
    )
    agent = ModelAgent(
        route="fake/model",
        policy_text=_POLICY,
        sampling=_SAMPLING,
        supports_response_format=False,
        completion_fn=fake,
        mandate_text=_MANDATE,
        reasoning=False,
    )
    obs = _make_observation()
    agent.act(obs)

    assert len(fake.recorded_kwargs) >= 1
    for kwargs in fake.recorded_kwargs:
        assert "reasoning_effort" not in kwargs
        assert "include_reasoning" not in kwargs
        assert "drop_params" not in kwargs
        # non-reasoning models keep the profile's max_tokens (no override)
        assert kwargs.get("max_tokens") == _SAMPLING.max_tokens


# ---------------------------------------------------------------------------
# Part 3 — null-content handling
# ---------------------------------------------------------------------------

def test_null_content_does_not_crash_and_returns_abort_malformed():
    """A completion returning content=None must not crash and must yield Abort(malformed_action).

    Specifically, the literal string "None" must never be emitted as an action.
    """
    fake = _NullContentFake()
    agent = ModelAgent(
        route="fake/model",
        policy_text=_POLICY,
        sampling=_SAMPLING,
        supports_response_format=False,
        completion_fn=fake,
        mandate_text=_MANDATE,
        reasoning=True,
    )
    obs = _make_observation()
    action = agent.act(obs)

    assert isinstance(action, Abort), f"Expected Abort, got {action!r}"
    assert action.reason == "malformed_action"
    # Both the initial call and the one reformat-retry must have fired (total == 2).
    assert fake.calls == 2, f"Expected 2 completion calls (initial + reformat retry), got {fake.calls}"
    # The transcript must not contain the literal string "None" as an action.
    assistant_turns = [m for m in agent.transcript if m["role"] == "assistant"]
    for turn in assistant_turns:
        assert turn["content"] != "None", "Literal 'None' string was emitted as action content"


# ---------------------------------------------------------------------------
# Part 3 / agent_factory — reasoning threaded through factory
# ---------------------------------------------------------------------------

def test_agent_factory_threads_reasoning_true():
    """agent_factory with reasoning=True model should produce agents that pass reasoning kwargs."""
    fake = _RecordingFake(
        content='{"tool": "abort", "args": {"reason": "malformed_action"}}'
    )
    model = _make_model(reasoning=True)
    factory = agent_factory(
        model,
        policy_text=_POLICY,
        sampling=_SAMPLING,
        completion_fn=fake,
        mandate_text=_MANDATE,
    )
    agent = factory()
    obs = _make_observation()
    agent.act(obs)

    first_kwargs = fake.recorded_kwargs[0]
    assert first_kwargs.get("reasoning_effort") == REASONING_EFFORT
    assert first_kwargs.get("include_reasoning") is True
    assert first_kwargs.get("drop_params") is True


def test_agent_factory_threads_reasoning_false():
    """agent_factory with reasoning=False model must produce agents without reasoning kwargs."""
    fake = _RecordingFake(
        content='{"tool": "abort", "args": {"reason": "malformed_action"}}'
    )
    model = _make_model(reasoning=False)
    factory = agent_factory(
        model,
        policy_text=_POLICY,
        sampling=_SAMPLING,
        completion_fn=fake,
        mandate_text=_MANDATE,
    )
    agent = factory()
    obs = _make_observation()
    agent.act(obs)

    for kwargs in fake.recorded_kwargs:
        assert "reasoning_effort" not in kwargs
        assert "include_reasoning" not in kwargs
        assert "drop_params" not in kwargs


# ---------------------------------------------------------------------------
# REASONING_EFFORT constant
# ---------------------------------------------------------------------------

def test_reasoning_effort_constant_is_high():
    assert REASONING_EFFORT == "high"
