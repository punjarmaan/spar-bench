from decimal import Decimal
import pytest
from pydantic import ValidationError
from spar.simulator.enums import ViolationClass
from spar.simulator.mandates import (
    IntentMandate, CartMandate, PaymentMandate, ScopeViolation,
)

def _intent(**over):
    base = dict(
        goal="buy one widget",
        amount_limit=Decimal("100.00"),
        currency="USD",
        merchant_constraint=["acme"],
        mcc_constraint=["5734"],
        allowed_instruments=["card_visa"],
        per_txn_max=Decimal("100.00"),
        daily_remaining=Decimal("500.00"),
        session_ttl_steps=20,
        single_use_or_recurring="single_use",
        time_window=None,
        revoked=False,
        human_present=True,
        conditions={},
    )
    base.update(over)
    return IntentMandate(**base)

def test_intent_mandate_carries_named_scoped_fields():
    m = _intent()
    assert m.amount_limit == Decimal("100.00")
    assert m.merchant_constraint == ["acme"]
    assert m.single_use_or_recurring == "single_use"

def test_intent_mandate_rejects_unknown_field():
    with pytest.raises(ValidationError):
        _intent(unexpected="x")

def test_cart_mandate_has_payee_and_currency():
    c = CartMandate(
        line_items=[{"sku": "w", "qty": 1}],
        payee="acme",
        currency="USD",
        subtotal=Decimal("90.00"),
        total=Decimal("99.00"),
    )
    assert c.payee == "acme"
    assert c.currency == "USD"

def test_payment_mandate_carries_charge_facts():
    p = PaymentMandate(
        instrument="card_visa",
        amount=Decimal("99.00"),
        currency="USD",
        idempotency_key="k1",
    )
    assert p.amount == Decimal("99.00")

def test_scope_violation_is_typed_by_violation_class():
    v = ScopeViolation(
        violation_class=ViolationClass.OVERSPEND,
        violated_field="amount_limit",
        limit=Decimal("100.00"),
        attempted=Decimal("120.00"),
    )
    assert v.violation_class is ViolationClass.OVERSPEND
    assert v.violated_field == "amount_limit"
