"""ModelAgent accumulates a verbatim per-turn record incl. reasoning + retry (enrichment T3)."""
from decimal import Decimal

from spar.eval.agent import ModelAgent
from spar.eval.profile import StageSampling
from spar.simulator.contract import (
    Observation, ObsCart, ObsContext, ObsMethod,
)
from spar.simulator.mandates import IntentMandate


def _obs() -> Observation:
    return Observation(
        mandate=IntentMandate(
            goal="buy a widget", amount_limit=Decimal("100"), currency="USD",
            human_present=True, conditions={}, per_txn_max=Decimal("200"),
            daily_remaining=Decimal("350"), merchant_constraint=["acme"], mcc_constraint=None,
            allowed_instruments=["visa"], session_ttl_steps=10,
            single_use_or_recurring="single_use", time_window=None,
        ),
        cart=ObsCart(line_items=[{"sku": "w1", "qty": 1}], subtotal=Decimal("90")),
        methods=[ObsMethod(acquirer_id="acq_a", methods=["visa"], geos=["US"],
                           advertised_fee_bps=200, observed_approval_band="high")],
        context=ObsContext(buyer_geo="US", elapsed_steps=0),
    )


def _fn_with_reasoning(items):
    """items: list of (content, reasoning) tuples returned in order."""
    seq = iter(items)

    def _fn(**kwargs):
        content, reasoning = next(seq)
        msg = type("M", (), {"content": content, "reasoning_content": reasoning})()
        choice = type("C", (), {"message": msg})()
        return type("R", (), {
            "choices": [choice],
            "usage": type("U", (), {"prompt_tokens": 100, "completion_tokens": 5})(),
            "_hidden_params": {"response_cost": 0.0009},
        })()

    return _fn


def test_turn_captures_observation_reasoning_and_action():
    fn = _fn_with_reasoning([('{"tool": "capture", "args": {}}', "I will capture")])
    agent = ModelAgent(
        route="r", policy_text="p", sampling=StageSampling(temperature=0.0),
        supports_response_format=False, completion_fn=fn, mandate_text="m",
    )
    agent.act(_obs())
    assert len(agent.turns) == 1
    turn = agent.turns[0]
    assert turn.index == 0
    assert turn.reasoning == "I will capture"
    # Normalized parsed action: {tool, args} where args is the model_dump minus the discriminator
    # (same convention as the original _write_trajectory; verbatim raw text lives in raw_output).
    assert turn.action["tool"] == "capture"
    assert isinstance(turn.action["args"], dict)
    assert turn.raw_output == '{"tool": "capture", "args": {}}'
    assert turn.observation["mandate"]["goal"] == "buy a widget"
    assert turn.retried is False
    assert len(turn.usage) == 1 and turn.usage[0].prompt_tokens == 100


def test_turn_captures_retry_path():
    fn = _fn_with_reasoning([
        ("not json", "first thoughts"),
        ('{"tool": "abort", "args": {"reason": "done"}}', "retry thoughts"),
    ])
    agent = ModelAgent(
        route="r", policy_text="p", sampling=StageSampling(temperature=0.0),
        supports_response_format=False, completion_fn=fn, mandate_text="m",
    )
    agent.act(_obs())
    turn = agent.turns[0]
    assert turn.retried is True
    assert turn.raw_output == "not json"
    assert turn.reasoning == "first thoughts"
    assert turn.retry_raw_output == '{"tool": "abort", "args": {"reason": "done"}}'
    assert turn.retry_reasoning == "retry thoughts"
    assert turn.action == {"tool": "abort", "args": {"reason": "done"}}
    assert len(turn.usage) == 2  # initial + retry call
