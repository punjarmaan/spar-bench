from decimal import Decimal

import pytest
from pydantic import ValidationError

from spar.simulator.mandates import (
    ScopedAuthority, IntentMandate, CartMandate, PaymentMandate, MarketContext, ScopeViolation,  # noqa: F401
)


def test_scoped_authority_defaults_not_revoked():
    a = ScopedAuthority(
        per_txn_max=Decimal("200"), daily_remaining=Decimal("350"),
        merchant_allowlist=["acme-store"], mcc_allowlist=None,
        allowed_instruments=["visa", "mc"], session_ttl_steps=20,
    )
    assert a.revoked is False


def test_intent_mandate_allows_null_ceiling_for_semantic():
    m = IntentMandate(
        goal="buy a not-too-expensive coffee maker", price_ceiling=None,
        budget_hint="not too expensive", currency="USD", human_present=False,
        conditions={"geo": "US"},
        authority=ScopedAuthority(
            per_txn_max=Decimal("200"), daily_remaining=Decimal("350"),
            merchant_allowlist=["acme"], mcc_allowlist=None,
            allowed_instruments=["visa"], session_ttl_steps=20,
        ),
    )
    assert m.price_ceiling is None and m.budget_hint == "not too expensive"


def test_money_must_not_be_float():
    with pytest.raises(ValidationError):
        ScopedAuthority(
            per_txn_max=200.5, daily_remaining=Decimal("1"), merchant_allowlist=[],
            mcc_allowlist=None, allowed_instruments=[], session_ttl_steps=1,
        )


def test_scope_violation_kind_is_constrained():
    v = ScopeViolation(kind="per_txn_max_exceeded", limit=Decimal("200"), attempted=Decimal("250"))
    assert v.kind == "per_txn_max_exceeded"
