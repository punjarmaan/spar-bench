from decimal import Decimal

from spar.simulator.enums import Axis, Difficulty, IntentSpec, FsmState
from spar.simulator.schemas import Acquirer, WorldConfig, Gold, Sample
from spar.simulator.mandates import IntentMandate, ScopedAuthority


def _authority() -> ScopedAuthority:
    return ScopedAuthority(
        per_txn_max=Decimal("200"), daily_remaining=Decimal("350"),
        merchant_allowlist=["acme"], mcc_allowlist=None,
        allowed_instruments=["visa"], session_ttl_steps=20,
    )


def test_acquirer_separates_exposed_and_hidden_fields():
    acq = Acquirer(
        acquirer_id="acq_a", methods=["visa"], supported_geos=["US"],
        advertised_fee_bps=250, observed_approval_band="med",
        true_fee_bps=300, approval_prob=0.8, reliability=0.95,
    )
    # Exposed vs hidden are both present on the server-side model; redaction happens in Observation.
    assert acq.advertised_fee_bps == 250 and acq.observed_approval_band == "med"
    assert acq.approval_prob == 0.8


def test_sample_round_trips_through_json():
    sample = Sample(
        sample_id="spar_decline_recovery_0001", axis=Axis.DECLINE_RECOVERY,
        difficulty=Difficulty.EASY, is_trap=False, intent_spec=IntentSpec.EXPLICIT,
        diamond=False, model_graded=False, seed=42, canary="spar:test-uuid",
        world_config=WorldConfig(acquirers=[], max_steps=30),
        mandate=IntentMandate(
            goal="buy widget", price_ceiling=Decimal("100"), currency="USD",
            human_present=True, conditions={}, authority=_authority(),
        ),
        policy_id="default_v1",
        gold=Gold(correct_outcome=FsmState.CLOSED, must=["compute_tax"], must_not=["retry_hard_decline"]),
    )
    restored = Sample.model_validate_json(sample.model_dump_json())
    assert restored == sample
    assert restored.gold.correct_outcome is FsmState.CLOSED
