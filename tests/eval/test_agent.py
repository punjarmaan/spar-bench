import json
from decimal import Decimal

from spar.agents.base import Agent
from spar.eval.agent import (
    SCAFFOLD_VERSION,
    CallUsage,
    ModelAgent,
    agent_factory,
    render_observation,
    tool_catalog,
)
from spar.eval.models import ModelConfig
from spar.eval.profile import StageSampling
from spar.simulator.contract import (
    Abort,
    Observation,
    ObsCart,
    ObsContext,
    ObsMethod,
    SelectRoute,
)
from spar.simulator.mandates import IntentMandate


def test_scaffold_version_is_pinned():
    assert SCAFFOLD_VERSION == "1.0.0"


def test_tool_catalog_is_json_covering_all_nine_tools():
    cat = tool_catalog()
    assert isinstance(cat, str)
    parsed = json.loads(cat)  # must be valid JSON so it never drifts from the contract
    # Catalog maps each tool name -> its JSON arg schema, derived from contract models.
    names = set(parsed)
    assert names == {
        "select_route", "compute_tax", "submit_authorization", "handle_challenge",
        "retry", "modify_cart", "request_user_confirmation", "capture", "abort",
    }
    # Args schema is derived from the pydantic models (so it cannot drift).
    assert "acquirer_id" in parsed["select_route"]["properties"]
    assert "method" in parsed["select_route"]["properties"]
    assert "reason" in parsed["abort"]["properties"]
    assert "strategy" in parsed["retry"]["properties"]
    # compute_tax / capture take no args beyond the discriminator.
    assert set(parsed["compute_tax"]["properties"]) == set()


def test_tool_catalog_is_deterministic():
    assert tool_catalog() == tool_catalog()


def _obs() -> Observation:
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


def test_render_observation_is_string_with_salient_fields():
    text = render_observation(_obs())
    assert isinstance(text, str)
    # Salient, agent-visible state appears so the model can act on it.
    assert "buy a widget" in text
    assert "acq_a" in text
    assert "visa" in text
    assert "US" in text


def test_render_observation_is_deterministic_and_json_parseable():
    text = render_observation(_obs())
    assert render_observation(_obs()) == text
    # Rendered as JSON of the observation (Decimal serialized as a string by pydantic json mode).
    parsed = json.loads(text)
    assert parsed["mandate"]["goal"] == "buy a widget"
    assert parsed["methods"][0]["acquirer_id"] == "acq_a"


def test_call_usage_holds_tokens_and_optional_cost():
    u = CallUsage(prompt_tokens=120, completion_tokens=8, response_cost=0.0021)
    assert u.prompt_tokens == 120
    assert u.completion_tokens == 8
    assert u.response_cost == 0.0021
    # cost is a float USD (never Decimal); response_cost is optional.
    assert isinstance(u.response_cost, float)
    u2 = CallUsage(prompt_tokens=1, completion_tokens=1, response_cost=None)
    assert u2.response_cost is None


def _fake_completion_factory(contents):
    """Returns a completion_fn that yields `contents` in order and records every call's kwargs."""
    calls = []
    seq = iter(contents)

    def _fn(**kwargs):
        calls.append(kwargs)
        content = next(seq)

        class _Msg:
            pass

        msg = _Msg()
        msg.content = content

        class _Choice:
            pass

        choice = _Choice()
        choice.message = msg

        class _Usage:
            prompt_tokens = 100
            completion_tokens = 5

        class _Resp:
            choices = [choice]
            usage = _Usage()
            _hidden_params = {"response_cost": 0.0009}

        return _Resp()

    return _fn, calls


def test_model_agent_satisfies_agent_protocol():
    fn, _ = _fake_completion_factory(['{"tool": "abort", "args": {"reason": "x"}}'])
    agent = ModelAgent(
        route="openrouter/test/m", policy_text="POLICY-TEXT", sampling=StageSampling(temperature=0.0),
        supports_response_format=True, completion_fn=fn, mandate_text="MANDATE-TEXT",
    )
    assert isinstance(agent, Agent)


def test_model_agent_builds_system_prompt_and_returns_parsed_action():
    fn, calls = _fake_completion_factory(
        ['{"tool": "select_route", "args": {"acquirer_id": "acq_a", "method": "visa"}}']
    )
    agent = ModelAgent(
        route="openrouter/test/m", policy_text="POLICY-DOC-XYZ",
        sampling=StageSampling(temperature=0.0, top_p=1.0, max_tokens=2048, seed=7),
        supports_response_format=True, completion_fn=fn, mandate_text="MANDATE-ABC",
    )
    action = agent.act(_obs())
    assert isinstance(action, SelectRoute)
    assert action.acquirer_id == "acq_a" and action.method == "visa"

    kw = calls[0]
    assert kw["model"] == "openrouter/test/m"
    # Sampling threaded straight from the StageSampling.
    assert kw["temperature"] == 0.0
    assert kw["top_p"] == 1.0
    assert kw["max_tokens"] == 2048
    assert kw["seed"] == 7
    # JSON mode requested because supports_response_format is True.
    assert kw["response_format"] == {"type": "json_object"}
    # System prompt (message 0) carries policy + mandate + tool catalog + the output contract.
    system = kw["messages"][0]
    assert system["role"] == "system"
    assert "POLICY-DOC-XYZ" in system["content"]
    assert "MANDATE-ABC" in system["content"]
    assert "select_route" in system["content"]            # the tool catalog
    assert '{"tool"' in system["content"]                 # the JSON output contract
    # The rendered observation is the user turn.
    assert kw["messages"][-1]["role"] == "user"
    assert "buy a widget" in kw["messages"][-1]["content"]


def test_model_agent_captures_usage_per_call():
    fn, _ = _fake_completion_factory(['{"tool": "capture", "args": {}}'])
    agent = ModelAgent(
        route="r", policy_text="p", sampling=StageSampling(temperature=0.0),
        supports_response_format=True, completion_fn=fn, mandate_text="m",
    )
    agent.act(_obs())
    assert len(agent.usage) == 1
    assert agent.usage[0].prompt_tokens == 100
    assert agent.usage[0].completion_tokens == 5
    assert agent.usage[0].response_cost == 0.0009


def test_model_agent_skips_json_mode_when_unsupported():
    fn, calls = _fake_completion_factory(['{"tool": "abort", "args": {"reason": "x"}}'])
    agent = ModelAgent(
        route="r", policy_text="p", sampling=StageSampling(temperature=0.0),
        supports_response_format=False, completion_fn=fn, mandate_text="m",
    )
    agent.act(_obs())
    assert "response_format" not in calls[0]


def test_model_agent_keeps_transcript_state_across_acts():
    fn, calls = _fake_completion_factory([
        '{"tool": "select_route", "args": {"acquirer_id": "acq_a", "method": "visa"}}',
        '{"tool": "submit_authorization", "args": {}}',
    ])
    agent = ModelAgent(
        route="r", policy_text="p", sampling=StageSampling(temperature=0.0),
        supports_response_format=True, completion_fn=fn, mandate_text="m",
    )
    agent.act(_obs())
    agent.act(_obs())
    # Second call's transcript grew: system + obs1 + assistant1 + obs2.
    assert len(calls[1]["messages"]) > len(calls[0]["messages"])
    # The model's prior reply is replayed as an assistant turn.
    assert any(m["role"] == "assistant" for m in calls[1]["messages"])


def test_model_agent_recovers_on_reformat_retry():
    # First reply is junk, retry returns a valid action.
    fn, calls = _fake_completion_factory([
        "here you go: not json at all",
        '{"tool": "abort", "args": {"reason": "done"}}',
    ])
    agent = ModelAgent(
        route="r", policy_text="p", sampling=StageSampling(temperature=0.0),
        supports_response_format=True, completion_fn=fn, mandate_text="m",
    )
    action = agent.act(_obs())
    assert isinstance(action, Abort)
    assert action.reason == "done"
    assert len(calls) == 2                                   # exactly one retry
    assert "EXACTLY ONE JSON" in calls[1]["messages"][-1]["content"]  # the reformat nudge


def test_model_agent_aborts_malformed_after_failed_retry():
    fn, calls = _fake_completion_factory(["garbage", "still garbage"])
    agent = ModelAgent(
        route="r", policy_text="p", sampling=StageSampling(temperature=0.0),
        supports_response_format=True, completion_fn=fn, mandate_text="m",
    )
    action = agent.act(_obs())
    assert isinstance(action, Abort)
    assert action.reason == "malformed_action"
    assert len(calls) == 2                                   # one call + one bounded retry, no more


def test_model_agent_unknown_tool_triggers_retry_then_abort():
    fn, calls = _fake_completion_factory([
        '{"tool": "teleport", "args": {}}',                  # not one of the 9 tools
        '{"tool": "frobnicate", "args": {}}',                # still unknown
    ])
    agent = ModelAgent(
        route="r", policy_text="p", sampling=StageSampling(temperature=0.0),
        supports_response_format=True, completion_fn=fn, mandate_text="m",
    )
    action = agent.act(_obs())
    assert isinstance(action, Abort)
    assert action.reason == "malformed_action"
    assert len(calls) == 2


def test_model_agent_bad_args_triggers_retry_then_abort():
    fn, calls = _fake_completion_factory([
        '{"tool": "select_route", "args": {"acquirer_id": "acq_a"}}',  # missing required `method`
        '{"tool": "select_route", "args": {}}',                        # still invalid
    ])
    agent = ModelAgent(
        route="r", policy_text="p", sampling=StageSampling(temperature=0.0),
        supports_response_format=True, completion_fn=fn, mandate_text="m",
    )
    action = agent.act(_obs())
    assert isinstance(action, Abort)
    assert action.reason == "malformed_action"
    assert len(calls) == 2


def test_agent_factory_produces_fresh_independent_model_agents():
    fn, _ = _fake_completion_factory([
        '{"tool": "abort", "args": {"reason": "a"}}',
        '{"tool": "abort", "args": {"reason": "b"}}',
    ])
    model = ModelConfig(id="m", route="openrouter/test/m", cls="open",
                        supports_response_format=False)
    factory = agent_factory(
        model, policy_text="POL", sampling=StageSampling(temperature=0.7),
        completion_fn=fn, mandate_text="MND",
    )
    a1 = factory()
    a2 = factory()
    assert isinstance(a1, ModelAgent) and isinstance(a2, ModelAgent)
    assert a1 is not a2                                      # independent trials (spec §5.1)
    # Identity/sampling threaded from the ModelConfig + StageSampling.
    assert a1.route == "openrouter/test/m"
    assert a1.sampling.temperature == 0.7
    assert a1.supports_response_format is False
    # Acting on one does not pollute the other's transcript/usage.
    a1.act(_obs())
    assert len(a1.usage) == 1 and len(a2.usage) == 0
    assert len(a2.transcript) == 1                          # just the system prompt
