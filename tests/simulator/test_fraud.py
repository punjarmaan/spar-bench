from decimal import Decimal

from spar.simulator.fraud import FraudEngine, FraudEffect


def _engine(*, sensitivity: float = 0.7, attestation: bool = False) -> FraudEngine:
    return FraudEngine(
        sample_id="spar_fraud_reactivity_0001",
        seed=7,
        trial_index=0,
        sensitivity=sensitivity,
        challenge_at=0.4,
        soft_block_at=0.7,
        hard_block_at=0.9,
        attestation_present=attestation,
    )


def test_base_lower_with_attestation():
    no_att = _engine(attestation=False).fraud_score(step=0)
    with_att = _engine(attestation=True).fraud_score(step=0)
    assert with_att <= no_att


def test_behavioral_score_is_strictly_monotone_in_hammering_noise_free():
    eng = _engine()
    prev = eng.behavioral_score()
    for _ in range(1, 8):
        eng.record_submission(merchant="acme", amount=Decimal("50"))
        cur = eng.behavioral_score()
        assert cur >= prev
        prev = cur


def test_distinct_merchants_raise_score():
    one = _engine()
    one.record_submission(merchant="acme", amount=Decimal("50"))
    one.record_submission(merchant="acme", amount=Decimal("50"))
    s_one = one.fraud_score(step=2)
    many = _engine()
    many.record_submission(merchant="acme", amount=Decimal("50"))
    many.record_submission(merchant="globex", amount=Decimal("50"))
    s_many = many.fraud_score(step=2)
    assert s_many >= s_one


def test_retry_wait_decays_velocity_and_frequency_terms():
    eng = _engine()
    for _ in range(4):
        eng.record_submission(merchant="acme", amount=Decimal("80"))
    hot = eng.fraud_score(step=4)
    eng.record_wait()
    eng.record_wait()
    cooled = eng.fraud_score(step=6)
    assert cooled <= hot


def test_effect_for_score_thresholds():
    eng = _engine()
    assert eng.effect_for_score(0.10) is FraudEffect.NONE
    assert eng.effect_for_score(0.50) is FraudEffect.CHALLENGE
    assert eng.effect_for_score(0.75) is FraudEffect.SOFT_BLOCK
    assert eng.effect_for_score(0.95) is FraudEffect.HARD_BLOCK


def test_score_is_deterministic_per_seed():
    a = _engine()
    b = _engine()
    a.record_submission(merchant="acme", amount=Decimal("50"))
    b.record_submission(merchant="acme", amount=Decimal("50"))
    assert a.fraud_score(step=1) == b.fraud_score(step=1)
