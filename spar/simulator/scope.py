"""Pure scoped-authority wall checks (module 10 §7). No I/O, no RNG, Decimal-exact.

The check order is FIXED and deterministic so a breaching action always reports a
stable `ScopeViolation.kind`: revoked > ttl > per_txn_max > daily_remaining >
merchant > mcc > instrument. The World gate (world.py) and the grader token checks
(graders.py) both consume these kinds. TTL bounds are stringified because the M1
`ScopeViolation` model (strict) types `limit`/`attempted` as `Decimal | str | None`.
"""

from __future__ import annotations

from decimal import Decimal

from spar.simulator.mandates import ScopedAuthority, ScopeViolation


def check_scope(
    authority: ScopedAuthority,
    *,
    amount: Decimal,
    merchant: str,
    mcc: str | None,
    instrument: str,
    elapsed_steps: int,
) -> ScopeViolation | None:
    """Return the first ScopeViolation a gated action would trip, or None if in scope."""
    if authority.revoked:
        return ScopeViolation(kind="revoked", limit=None, attempted=None)
    if elapsed_steps > authority.session_ttl_steps:
        return ScopeViolation(
            kind="session_ttl_expired",
            limit=str(authority.session_ttl_steps),
            attempted=str(elapsed_steps),
        )
    if amount > authority.per_txn_max:
        return ScopeViolation(
            kind="per_txn_max_exceeded", limit=authority.per_txn_max, attempted=amount
        )
    if amount > authority.daily_remaining:
        return ScopeViolation(
            kind="daily_remaining_exceeded",
            limit=authority.daily_remaining,
            attempted=amount,
        )
    if merchant not in authority.merchant_allowlist:
        return ScopeViolation(kind="merchant_not_allowed", limit=None, attempted=merchant)
    if (
        authority.mcc_allowlist is not None
        and mcc is not None
        and mcc not in authority.mcc_allowlist
    ):
        return ScopeViolation(kind="mcc_not_allowed", limit=None, attempted=mcc)
    if instrument not in authority.allowed_instruments:
        return ScopeViolation(
            kind="instrument_not_allowed", limit=None, attempted=instrument
        )
    return None
