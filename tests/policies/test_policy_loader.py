from spar.policies.loader import load_policy, KNOWN_POLICIES


def test_default_v1_is_known():
    assert "default_v1" in KNOWN_POLICIES


def test_load_policy_returns_prose_with_required_rules():
    text = load_policy("default_v1")
    lowered = text.lower()
    # The prose must state every machine-enforced rule so policy <-> gold can agree.
    assert "per_txn_max" in lowered
    assert "daily_remaining" in lowered
    assert "merchant_allowlist" in lowered or "allowlist" in lowered
    assert "mcc" in lowered
    assert "allowed_instruments" in lowered or "instrument" in lowered
    assert "restricted" in lowered
    assert "revoc" in lowered  # revocation / revoked
    assert "session_ttl" in lowered or "ttl" in lowered
    assert "request_user_confirmation" in lowered
    assert "human_present" in lowered
    assert "injected" in lowered or "injection" in lowered  # injection etiquette
    assert len(text) > 800  # a real rulebook, not a stub


def test_load_unknown_policy_raises():
    import pytest
    with pytest.raises(ValueError):
        load_policy("nope_v9")
