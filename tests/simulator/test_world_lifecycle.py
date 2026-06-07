from spar.simulator.contract import (
    Capture,
    ComputeTax,
    Refund,
    SelectRoute,
    SubmitAuthorization,
    Void,
)
from spar.simulator.enums import FsmState, ToolStatus
from tests.simulator._world_fixtures import build_world


def _route_and_auth(w, key="k1"):
    w.step(SelectRoute(tool="select_route", acquirer_id="a1", method="card_visa"))
    return w.step(SubmitAuthorization(tool="submit_authorization", idempotency_key=key))


def test_void_after_capture_returns_illegal():
    w = build_world()
    _route_and_auth(w)
    w.step(Capture(tool="capture", idempotency_key="k2"))
    resp = w.step(Void(tool="void", idempotency_key="k3"))
    assert resp.status is ToolStatus.ILLEGAL_ACTION


def test_reused_capture_key_does_not_double_charge():
    w = build_world()
    _route_and_auth(w)
    r1 = w.step(Capture(tool="capture", idempotency_key="k2"))
    r2 = w.step(Capture(tool="capture", idempotency_key="k2"))  # replay
    assert r2.status == r1.status
    assert w.hidden_final_state.get("duplicate_charge") is not True


def test_reused_key_replays_full_response_not_just_status():
    w = build_world()
    w.step(SelectRoute(tool="select_route", acquirer_id="a1", method="card_visa"))
    r1 = w.step(SubmitAuthorization(tool="submit_authorization", idempotency_key="k1"))
    r2 = w.step(SubmitAuthorization(tool="submit_authorization", idempotency_key="k1"))
    assert r2 == r1  # full ToolResponse equality (loss-free round-trip)


def test_txn_ordinal_stable_across_reused_key_replay():
    w = build_world()
    _route_and_auth(w, key="k1")
    before = w._txn_ordinal
    w.step(SubmitAuthorization(tool="submit_authorization", idempotency_key="k1"))  # replay
    assert w._txn_ordinal == before


def test_fresh_key_recapture_on_pending_intent_flags_duplicate():
    # Latency 5 keeps the world in CAPTURE_PENDING across the second capture so the fresh-key
    # re-capture on the same intent reaches the duplicate-detection block.
    w = build_world(settlement="async", decline_plan={"capture_latency_steps": 5})
    _route_and_auth(w)
    w.step(Capture(tool="capture", idempotency_key="k2"))  # CAPTURE_PENDING
    w.step(Capture(tool="capture", idempotency_key="k3"))  # fresh key, same intent
    assert w.hidden_final_state.get("duplicate_charge") is True


def test_lifecycle_and_world_state_agree_after_void():
    w = build_world()
    _route_and_auth(w)  # APPROVED
    w.step(Void(tool="void", idempotency_key="k2"))
    assert w.state is FsmState.VOIDED and w.lifecycle.state is FsmState.VOIDED


def test_lifecycle_and_world_state_agree_after_refund():
    w = build_world()  # sync settle, no refund window
    _route_and_auth(w)
    w.step(Capture(tool="capture", idempotency_key="k2"))  # SETTLED, lifecycle CAPTURED
    w.step(Refund(tool="refund", idempotency_key="k3"))
    assert w.state is FsmState.REFUNDED and w.lifecycle.state is FsmState.REFUNDED


def test_txn_ordinal_path_independent_across_action_counts():
    # Reaching the same logical (acq, method, amount) via different legal action counts
    # yields the same _txn_ordinal.
    a = build_world()
    _route_and_auth(a)
    b = build_world()
    b.step(SelectRoute(tool="select_route", acquirer_id="a1", method="card_visa"))
    b.step(ComputeTax(tool="compute_tax"))  # extra legal action, not a new intent
    b.step(SubmitAuthorization(tool="submit_authorization", idempotency_key="k1"))
    assert a._txn_ordinal == b._txn_ordinal == 1
