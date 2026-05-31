from decimal import Decimal
import pytest
from spar.simulator.enums import FsmState
from spar.simulator.lifecycle import Lifecycle, LifecycleError

def test_void_of_authorization_is_legal():
    lc = Lifecycle(txn_ordinal=0)
    lc.authorize(amount=Decimal("10"))
    lc.void()
    assert lc.state is FsmState.VOIDED

def test_void_after_capture_is_illegal():
    lc = Lifecycle(txn_ordinal=0)
    lc.authorize(amount=Decimal("10"))
    lc.capture()
    with pytest.raises(LifecycleError):
        lc.void()

def test_refund_before_capture_is_illegal():
    lc = Lifecycle(txn_ordinal=0)
    lc.authorize(amount=Decimal("10"))
    with pytest.raises(LifecycleError):
        lc.refund(post_capture_ordinal=1)

def test_refund_inside_window_is_legal():
    lc = Lifecycle(txn_ordinal=0, refund_window_len=3)
    lc.authorize(amount=Decimal("10"))
    lc.capture()
    lc.refund(post_capture_ordinal=2)
    assert lc.state is FsmState.REFUNDED

def test_refund_after_window_is_illegal_and_flags_missed_window():
    lc = Lifecycle(txn_ordinal=0, refund_window_len=3)
    lc.authorize(amount=Decimal("10"))
    lc.capture()
    with pytest.raises(LifecycleError):
        lc.refund(post_capture_ordinal=4)
    assert lc.missed_reversal_window is True

def test_authorize_only_from_cart_or_route_selected():
    lc = Lifecycle(txn_ordinal=0)
    lc.authorize(amount=Decimal("10"))
    # second authorize from APPROVED is illegal
    with pytest.raises(LifecycleError):
        lc.authorize(amount=Decimal("20"))

def test_refund_at_exactly_window_len_is_legal():
    # boundary: ordinal == refund_window_len is still inside the window (> is the cutoff)
    lc = Lifecycle(txn_ordinal=0, refund_window_len=3)
    lc.authorize(amount=Decimal("10"))
    lc.capture()
    lc.refund(post_capture_ordinal=3)
    assert lc.state is FsmState.REFUNDED
