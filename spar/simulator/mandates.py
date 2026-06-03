"""Three-mandate consent layer (Intent / Cart / Payment)."""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from spar.simulator.enums import ViolationClass


class _Strict(BaseModel):
    # strict=True keeps money exact: a float is NEVER silently coerced into a Decimal field.
    model_config = ConfigDict(strict=True, extra="forbid")


class IntentMandate(_Strict):
    """What the user authorized. The named scoped fields are the enforcement primitives
    every out-of-scope trap violates."""

    goal: str
    # The consent constraints:
    amount_limit: Decimal | None  # None for semantic / underspecified intents
    currency: str
    merchant_constraint: list[str]
    mcc_constraint: list[str] | None
    allowed_instruments: list[str]
    per_txn_max: Decimal
    daily_remaining: Decimal
    session_ttl_steps: int
    single_use_or_recurring: Literal["single_use", "recurring"]
    time_window: tuple[int, int] | None  # (open_step, close_step) in transition ordinals
    revoked: bool = False
    budget_hint: str | None = None
    human_present: bool = True
    conditions: dict[str, Any] = {}


class CartMandate(_Strict):
    """What the agent assembled / proposes to pay. v1 enforces payee + currency."""

    line_items: list[dict[str, Any]]
    payee: str
    currency: str
    subtotal: Decimal
    total: Decimal | None = None


class PaymentMandate(_Strict):
    """What is actually charged."""

    instrument: str
    amount: Decimal
    currency: str
    idempotency_key: str | None = None


class ScopeViolation(_Strict):
    """A single consent breach, typed by catastrophic class + the named field it broke."""

    violation_class: ViolationClass
    violated_field: str
    limit: Decimal | str | None = None
    attempted: Decimal | str | None = None


class MarketContext(_Strict):
    """HIDDEN semantic-budget ground truth. Never exposed to the agent."""

    category: str
    currency: str
    p25: Decimal
    p50: Decimal
    p75: Decimal
    p90: Decimal
