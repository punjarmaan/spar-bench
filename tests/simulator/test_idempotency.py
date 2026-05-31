import pytest
from spar.simulator.idempotency import IdempotencyLedger

def test_fresh_key_is_a_miss():
    led = IdempotencyLedger()
    assert led.lookup("capture", "k1") is None

def test_reused_key_replays_cached_result():
    led = IdempotencyLedger()
    led.record("capture", "k1", {"status": "APPROVED"})
    assert led.lookup("capture", "k1") == {"status": "APPROVED"}

def test_same_tool_different_keys_are_independent():
    led = IdempotencyLedger()
    led.record("capture", "k1", {"status": "APPROVED"})
    assert led.lookup("capture", "k2") is None  # the duplicate-charge door

def test_missing_key_is_never_cached_and_always_a_miss():
    # Empty/None key = treated as fresh every time (spec §4.5): unsafe path stays reachable.
    led = IdempotencyLedger()
    assert led.lookup("capture", None) is None
    led.record("capture", None, {"status": "APPROVED"})
    assert led.lookup("capture", None) is None
    assert led.lookup("capture", "") is None

def test_record_idempotent_on_match_raises_on_conflict():
    led = IdempotencyLedger()
    led.record("capture", "k1", {"status": "APPROVED"})
    # idempotent: recording the SAME payload again is a no-op, NOT an error
    led.record("capture", "k1", {"status": "APPROVED"})
    assert led.lookup("capture", "k1") == {"status": "APPROVED"}
    # conflict: same key, DIFFERENT payload -> ValueError
    with pytest.raises(ValueError):
        led.record("capture", "k1", {"status": "DECLINED"})

def test_complex_payload_round_trips_loss_free():
    led = IdempotencyLedger()
    payload = {"status": "approved", "detail": {"amount": "99.00", "codes": ["a", "b"]}, "n": 3}
    led.record("submit_authorization", "k9", payload)
    assert led.lookup("submit_authorization", "k9") == payload
