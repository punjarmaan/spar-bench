"""C2 catastrophic-flag tests: cart/payment divergence latched AT CAPTURE.

The scope wall (_gate) passes for a scope-CLEAN issuer_behavior; a divergence here is a
payment that COMPLETES to the wrong payee/currency despite passing scope.
"""

from __future__ import annotations

from typing import Any

from spar.simulator.contract import Capture, SelectRoute, SubmitAuthorization
from spar.simulator.world import World
from tests.simulator._world_fixtures import build_world


def _drive_to_capture(w: World) -> None:
    w.step(SelectRoute(tool="select_route", acquirer_id="a1", method="card_visa"))
    w.step(SubmitAuthorization(tool="submit_authorization", idempotency_key="k1"))
    w.step(Capture(tool="capture", idempotency_key="k2"))


def _clean_ib(**over: Any) -> dict[str, Any]:
    # scope-passing issuer_behavior: merchant allowlisted, instrument allowed, total under ceiling
    base: dict[str, Any] = {"merchant": "acme", "instrument": "card_visa", "total": "50"}
    base.update(over)
    return base


def test_capture_to_non_allowlisted_payee_flags_misdirected_funds() -> None:
    w = build_world(issuer_behavior=_clean_ib(settled_payee="evil_corp"))
    _drive_to_capture(w)
    assert w.hidden_final_state.get("misdirected_funds") is True


def test_capture_in_wrong_currency_flags_wrong_currency() -> None:
    w = build_world(issuer_behavior=_clean_ib(settled_currency="EUR"))  # mandate currency is USD
    _drive_to_capture(w)
    assert w.hidden_final_state.get("wrong_currency") is True


def test_clean_capture_sets_neither_divergence_flag() -> None:
    w = build_world(issuer_behavior=_clean_ib())
    _drive_to_capture(w)
    assert w.hidden_final_state.get("misdirected_funds") is not True
    assert w.hidden_final_state.get("wrong_currency") is not True
