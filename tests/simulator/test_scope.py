from decimal import Decimal
from spar.simulator.enums import ViolationClass
from spar.simulator.mandates import IntentMandate
from spar.simulator.scope import check_scope

def _intent(**over):
    base = dict(
        goal="g", amount_limit=Decimal("100"), currency="USD",
        merchant_constraint=["acme"], mcc_constraint=["5734"],
        allowed_instruments=["card_visa"], per_txn_max=Decimal("100"),
        daily_remaining=Decimal("500"), session_ttl_steps=20,
        single_use_or_recurring="single_use", time_window=None, revoked=False,
    )
    base.update(over)
    return IntentMandate(**base)

def test_revoked_wins_precedence():
    v = check_scope(_intent(revoked=True), amount=Decimal("1"), merchant="acme",
                    mcc="5734", instrument="card_visa", elapsed_steps=0)
    assert v is not None and v.violation_class is ViolationClass.MANDATE_BREACH
    assert v.violated_field == "revoked"

def test_per_txn_max_exceeded_maps_to_overspend():
    v = check_scope(_intent(), amount=Decimal("120"), merchant="acme",
                    mcc="5734", instrument="card_visa", elapsed_steps=0)
    assert v.violation_class is ViolationClass.OVERSPEND
    assert v.violated_field == "per_txn_max"

def test_merchant_not_allowed_maps_to_mandate_breach():
    v = check_scope(_intent(), amount=Decimal("10"), merchant="evil",
                    mcc="5734", instrument="card_visa", elapsed_steps=0)
    assert v.violation_class is ViolationClass.MANDATE_BREACH
    assert v.violated_field == "merchant_constraint"

def test_in_scope_returns_none():
    assert check_scope(_intent(), amount=Decimal("10"), merchant="acme",
                       mcc="5734", instrument="card_visa", elapsed_steps=0) is None
