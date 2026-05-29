from decimal import Decimal

from spar.simulator.mandates import ScopedAuthority, ScopeViolation
from spar.simulator.scope import check_scope


def _auth(**overrides) -> ScopedAuthority:
    base = dict(
        per_txn_max=Decimal("200"), daily_remaining=Decimal("350"),
        merchant_allowlist=["acme"], mcc_allowlist=["5411"],
        allowed_instruments=["visa"], session_ttl_steps=20, revoked=False,
    )
    base.update(overrides)
    return ScopedAuthority(**base)


def test_in_scope_returns_none():
    v = check_scope(_auth(), amount=Decimal("100"), merchant="acme", mcc="5411",
                    instrument="visa", elapsed_steps=1)
    assert v is None


def test_per_txn_max_exceeded():
    v = check_scope(_auth(), amount=Decimal("250"), merchant="acme", mcc="5411",
                    instrument="visa", elapsed_steps=1)
    assert isinstance(v, ScopeViolation) and v.kind == "per_txn_max_exceeded"
    assert v.limit == Decimal("200") and v.attempted == Decimal("250")


def test_daily_remaining_exceeded_takes_priority_when_both_breach():
    v = check_scope(_auth(per_txn_max=Decimal("500")), amount=Decimal("400"),
                    merchant="acme", mcc="5411", instrument="visa", elapsed_steps=1)
    assert v.kind == "daily_remaining_exceeded"
    assert v.limit == Decimal("350") and v.attempted == Decimal("400")


def test_merchant_not_allowed():
    v = check_scope(_auth(), amount=Decimal("100"), merchant="evilcorp", mcc="5411",
                    instrument="visa", elapsed_steps=1)
    assert v.kind == "merchant_not_allowed" and v.attempted == "evilcorp"


def test_mcc_not_allowed_only_when_allowlist_present():
    v = check_scope(_auth(), amount=Decimal("100"), merchant="acme", mcc="7995",
                    instrument="visa", elapsed_steps=1)
    assert v.kind == "mcc_not_allowed" and v.attempted == "7995"
    ok = check_scope(_auth(mcc_allowlist=None), amount=Decimal("100"), merchant="acme",
                     mcc="7995", instrument="visa", elapsed_steps=1)
    assert ok is None


def test_instrument_not_allowed():
    v = check_scope(_auth(), amount=Decimal("100"), merchant="acme", mcc="5411",
                    instrument="amex", elapsed_steps=1)
    assert v.kind == "instrument_not_allowed" and v.attempted == "amex"


def test_session_ttl_expired_when_elapsed_exceeds_ttl():
    v = check_scope(_auth(session_ttl_steps=5), amount=Decimal("100"), merchant="acme",
                    mcc="5411", instrument="visa", elapsed_steps=6)
    # TTL bounds are stringified (ScopeViolation strict types reject bare int).
    assert v.kind == "session_ttl_expired" and v.limit == "5" and v.attempted == "6"


def test_revoked_takes_top_priority():
    v = check_scope(_auth(revoked=True), amount=Decimal("100"), merchant="acme",
                    mcc="5411", instrument="visa", elapsed_steps=1)
    assert v.kind == "revoked"
