from decimal import Decimal

from spar.simulator.contract import (
    Observation, ObsContext, ObsCart, ObsMethod, Action, ToolResponse, parse_action,  # noqa: F401
)
from spar.simulator.enums import ToolStatus


def test_action_tagged_union_parses_by_tool():
    a = parse_action({"tool": "retry", "strategy": "wait"})
    assert a.tool == "retry" and a.strategy == "wait"
    b = parse_action({"tool": "select_route", "acquirer_id": "acq_a", "method": "visa"})
    assert b.tool == "select_route" and b.method == "visa"


def test_tool_response_reason_code_only_on_declined():
    ok = ToolResponse(status=ToolStatus.DECLINED, reason_code="05")
    assert ok.reason_code == "05" and ok.challenge_token is None


def test_observation_context_carries_attempt_history():
    # F8: stateless agents need attempt history in context.
    ctx = ObsContext(buyer_geo="US", elapsed_steps=2, attempt_counts={"05": 1}, retry_count=1,
                     recent_events=["declined:05"])
    assert ctx.attempt_counts["05"] == 1 and ctx.retry_count == 1


def test_observation_alternatives_optional():
    cart = ObsCart(line_items=[], subtotal=Decimal("10"), computed_tax=None, total=None,
                   alternatives=[{"label": "a", "price": Decimal("38")}])
    assert cart.alternatives[0]["label"] == "a"
