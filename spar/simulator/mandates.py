"""AP2-aligned mandate models and the scoped-authority wall (module 10 §7)."""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


class _Strict(BaseModel):
    # Reject floats coerced into Decimal fields and unknown keys; money stays exact.
    model_config = ConfigDict(strict=True, extra="forbid")


class ScopedAuthority(_Strict):
    """ACP / Mastercard-Agentic-Token analog: the machine-enforced spending wall."""

    per_txn_max: Decimal
    daily_remaining: Decimal
    merchant_allowlist: list[str]
    mcc_allowlist: list[str] | None
    allowed_instruments: list[str]
    session_ttl_steps: int
    revoked: bool = False


class IntentMandate(_Strict):
    """AP2 intent mandate — the episode's rules of engagement."""

    goal: str
    price_ceiling: Decimal | None  # None for semantic / underspecified intents
    budget_hint: str | None = None  # free-text semantic budget when price_ceiling is None
    currency: str
    human_present: bool
    conditions: dict[str, Any]
    authority: ScopedAuthority


class CartMandate(_Strict):
    """AP2 cart mandate — the finalized items/price for a specific checkout."""

    line_items: list[dict[str, Any]]
    total: Decimal
    currency: str


class PaymentMandate(_Strict):
    """AP2 payment mandate — binds a specific instrument."""

    instrument: str
    network: str


class MarketContext(_Strict):
    """HIDDEN semantic-budget ground truth (module 10 §3.1). Never exposed to the agent."""

    category: str
    currency: str
    p25: Decimal
    p50: Decimal
    p75: Decimal
    p90: Decimal


class ScopeViolation(_Strict):
    kind: Literal[
        "per_txn_max_exceeded", "daily_remaining_exceeded",
        "merchant_not_allowed", "mcc_not_allowed",
        "instrument_not_allowed", "session_ttl_expired", "revoked",
    ]
    limit: Decimal | str | None = None
    attempted: Decimal | str | None = None
