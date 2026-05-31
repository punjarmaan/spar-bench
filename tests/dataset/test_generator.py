from decimal import Decimal

from spar.simulator.enums import Axis, Difficulty, FsmState, IntentSpec
from spar.simulator.schemas import Sample
from spar.dataset.generator import generate, GenSpec


def test_generate_returns_valid_sample_with_derived_gold():
    s = generate(GenSpec(axis=Axis.ROUTING, seed=1, difficulty=Difficulty.MEDIUM,
                         is_trap=False, intent_spec=IntentSpec.EXPLICIT))
    assert isinstance(s, Sample)
    assert s.axis is Axis.ROUTING
    assert s.gold.oracle_route is not None  # routing gold names the oracle route
    assert s.gold.correct_outcome is FsmState.CLOSED


def test_byte_identical_for_identical_inputs():
    spec = GenSpec(axis=Axis.DECLINE_RECOVERY, seed=99, difficulty=Difficulty.HARD,
                   is_trap=True, intent_spec=IntentSpec.EXPLICIT)
    a = generate(spec).model_dump_json()
    b = generate(spec).model_dump_json()
    assert a == b


def test_trap_sample_has_non_closed_gold():
    s = generate(GenSpec(axis=Axis.CONSENT_MANDATE, seed=3, difficulty=Difficulty.MEDIUM,
                         is_trap=True, intent_spec=IntentSpec.EXPLICIT))
    assert s.is_trap is True
    assert s.gold.correct_outcome in {FsmState.ABORTED, FsmState.ESCALATED}


def test_semantic_sample_derives_band_and_ceiling():
    s = generate(GenSpec(axis=Axis.CONSENT_MANDATE, seed=5, difficulty=Difficulty.MEDIUM,
                         is_trap=False, intent_spec=IntentSpec.SEMANTIC))
    assert s.mandate.amount_limit is None
    assert s.mandate.budget_hint is not None
    assert s.world_config.market_context is not None
    assert s.gold.acceptable_spend_band is not None
    lo, hi = s.gold.acceptable_spend_band
    assert lo == Decimal("0") and hi > Decimal("0")
    assert s.gold.overspend_ceiling is not None


def test_underspecified_sample_escalates_and_requires_confirmation():
    s = generate(GenSpec(axis=Axis.CONSENT_MANDATE, seed=8, difficulty=Difficulty.HARD,
                         is_trap=False, intent_spec=IntentSpec.UNDERSPECIFIED))
    assert s.gold.correct_outcome is FsmState.ESCALATED
    assert "request_user_confirmation" in s.gold.must
    assert s.gold.acceptable_spend_band is None


def test_non_trap_explicit_has_a_non_trivial_must():
    # F1: every non-trap requires >=1 non-trivial correct action.
    s = generate(GenSpec(axis=Axis.COMPLIANCE_TAX, seed=2, difficulty=Difficulty.EASY,
                         is_trap=False, intent_spec=IntentSpec.EXPLICIT))
    assert s.gold.correct_outcome is FsmState.CLOSED
    assert len(s.gold.must) >= 1


def test_money_fields_are_decimal():
    s = generate(GenSpec(axis=Axis.ROUTING, seed=1, difficulty=Difficulty.EASY,
                         is_trap=False, intent_spec=IntentSpec.EXPLICIT))
    assert isinstance(s.mandate.per_txn_max, Decimal)
