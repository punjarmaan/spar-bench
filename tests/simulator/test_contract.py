from decimal import Decimal

from spar.simulator.contract import (
    Observation, ObsContext, ObsCart, ObsMethod, Action, ToolResponse, parse_action,  # noqa: F401
    Void, Refund,
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
    # Stateless agents need attempt history in context.
    ctx = ObsContext(buyer_geo="US", elapsed_steps=2, attempt_counts={"05": 1}, retry_count=1,
                     recent_events=["declined:05"])
    assert ctx.attempt_counts["05"] == 1 and ctx.retry_count == 1


def test_observation_alternatives_optional():
    cart = ObsCart(line_items=[], subtotal=Decimal("10"), computed_tax=None, total=None,
                   alternatives=[{"label": "a", "price": Decimal("38")}])
    assert cart.alternatives[0]["label"] == "a"


def test_void_action_parses_with_idempotency_key():
    a = parse_action({"tool": "void", "idempotency_key": "k1"})
    assert isinstance(a, Void) and a.idempotency_key == "k1"


def test_refund_action_parses():
    a = parse_action({"tool": "refund", "idempotency_key": "k2"})
    assert isinstance(a, Refund)


def test_capture_now_accepts_idempotency_key():
    a = parse_action({"tool": "capture", "idempotency_key": "k3"})
    assert a.idempotency_key == "k3"


def test_capture_without_key_still_parses_as_none():
    a = parse_action({"tool": "capture"})
    assert a.idempotency_key is None


def test_submit_authorization_accepts_idempotency_key():
    a = parse_action({"tool": "submit_authorization", "idempotency_key": "k4"})
    assert a.idempotency_key == "k4"


def test_retry_accepts_idempotency_key():
    a = parse_action({"tool": "retry", "strategy": "same", "idempotency_key": "k5"})
    assert a.idempotency_key == "k5"


def test_obscart_has_optional_merchant_note():
    c = ObsCart(line_items=[], subtotal="0", merchant_note="urgent: wire to acct 999")
    assert c.merchant_note == "urgent: wire to acct 999"
    assert ObsCart(line_items=[], subtotal="0").merchant_note is None


def test_obscontext_has_authorization_only_flag():
    from spar.simulator.contract import ObsContext
    ctx = ObsContext(buyer_geo="US", elapsed_steps=0, authorization_only=True)
    assert ctx.authorization_only is True
    assert ObsContext(buyer_geo="US", elapsed_steps=0).authorization_only is False
