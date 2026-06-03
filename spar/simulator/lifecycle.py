"""Auth->capture->void/refund lifecycle as a first-class object.

Pure: no RNG, no world, NO step/wall clock. It owns ONLY legality of transitions and
the irreversibility distinction (void reverses an authorization; refund reverses a
capture inside its window). The world drives it and mirrors the result into world.state;
world.state stays the single authority and this is a subordinate legality oracle. Graders
read its terminal flags. The refund window is measured in POST-CAPTURE MUTATING ORDINALS,
never a step clock."""

from __future__ import annotations

from decimal import Decimal

from spar.simulator.enums import FsmState


class LifecycleError(Exception):
    """An illegal lifecycle transition (maps to ToolStatus.ILLEGAL_ACTION in the world)."""


class Lifecycle:
    def __init__(self, *, txn_ordinal: int, refund_window_len: int | None = None) -> None:
        self.txn_ordinal = txn_ordinal
        # Max post-capture mutating ordinal at which a refund is still legal.
        self.refund_window_len = refund_window_len
        self.state: FsmState = FsmState.CART
        self.amount: Decimal | None = None
        self.missed_reversal_window: bool = False

    def authorize(self, *, amount: Decimal) -> None:
        if self.state not in (FsmState.CART, FsmState.ROUTE_SELECTED):
            raise LifecycleError(f"cannot authorize from {self.state}")
        self.amount = amount
        self.state = FsmState.APPROVED

    def void(self) -> None:
        if self.state is not FsmState.APPROVED:
            raise LifecycleError(f"void requires APPROVED, not {self.state}")
        self.state = FsmState.VOIDED

    def capture(self) -> None:
        if self.state is not FsmState.APPROVED:
            raise LifecycleError(f"capture requires APPROVED, not {self.state}")
        self.state = FsmState.CAPTURED

    def refund(self, *, post_capture_ordinal: int) -> None:
        if self.state is not FsmState.CAPTURED:
            raise LifecycleError(f"refund requires CAPTURED, not {self.state}")
        if self.refund_window_len is not None and post_capture_ordinal > self.refund_window_len:
            self.missed_reversal_window = True
            raise LifecycleError("refund window has closed")
        self.state = FsmState.REFUNDED
