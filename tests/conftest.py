from decimal import Decimal

from spar.simulator.mandates import IntentMandate


def make_intent(**over) -> IntentMandate:
    base = dict(
        goal="g", amount_limit=Decimal("100"), currency="USD",
        merchant_constraint=["acme"], mcc_constraint=["5734"],
        allowed_instruments=["card_visa"], per_txn_max=Decimal("100"),
        daily_remaining=Decimal("500"), session_ttl_steps=20,
        single_use_or_recurring="single_use", time_window=None, revoked=False,
    )
    base.update(over)
    return IntentMandate(**base)
