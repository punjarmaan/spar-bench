"""Scoped-authority wall checks against named IntentMandate fields.

Precedence: revoked > session_ttl > per_txn_max > daily_remaining > merchant > mcc >
instrument.
"""

from __future__ import annotations

from decimal import Decimal

from spar.simulator.enums import ViolationClass
from spar.simulator.mandates import IntentMandate, ScopeViolation


def check_scope(
    mandate: IntentMandate,
    *,
    amount: Decimal,
    merchant: str,
    mcc: str | None,
    instrument: str,
    elapsed_steps: int,
) -> ScopeViolation | None:
    """Return the first ScopeViolation a gated action would trip, or None if in scope."""
    if mandate.revoked:
        return ScopeViolation(
            violation_class=ViolationClass.MANDATE_BREACH, violated_field="revoked"
        )
    if elapsed_steps > mandate.session_ttl_steps:
        return ScopeViolation(
            violation_class=ViolationClass.MANDATE_BREACH,
            violated_field="session_ttl_steps",
            limit=str(mandate.session_ttl_steps),
            attempted=str(elapsed_steps),
        )
    if amount > mandate.per_txn_max:
        return ScopeViolation(
            violation_class=ViolationClass.OVERSPEND, violated_field="per_txn_max",
            limit=mandate.per_txn_max, attempted=amount,
        )
    if amount > mandate.daily_remaining:
        return ScopeViolation(
            violation_class=ViolationClass.OVERSPEND, violated_field="daily_remaining",
            limit=mandate.daily_remaining, attempted=amount,
        )
    if merchant not in mandate.merchant_constraint:
        return ScopeViolation(
            violation_class=ViolationClass.MANDATE_BREACH,
            violated_field="merchant_constraint", attempted=merchant,
        )
    if mandate.mcc_constraint is not None and mcc is not None and mcc not in mandate.mcc_constraint:
        return ScopeViolation(
            violation_class=ViolationClass.MANDATE_BREACH,
            violated_field="mcc_constraint", attempted=mcc,
        )
    if instrument not in mandate.allowed_instruments:
        return ScopeViolation(
            violation_class=ViolationClass.MANDATE_BREACH,
            violated_field="allowed_instruments", attempted=instrument,
        )
    return None
