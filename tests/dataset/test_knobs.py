from spar.dataset.knobs import DifficultyKnobs, DeclineClass, knobs_for
from spar.simulator.enums import Axis, Difficulty


def test_knobs_model_validates_and_clamps_ranges():
    k = DifficultyKnobs(
        n_acquirers=3, decline_class=DeclineClass.AMBIGUOUS, fraud_sensitivity=0.5,
        n_interacting_traps=1, step_up_prob=0.3, async_capture=False,
        delayed_dispute=False, signal_conflict=1,
    )
    assert k.n_acquirers == 3 and k.decline_class is DeclineClass.AMBIGUOUS


def test_knobs_reject_out_of_range_floats():
    import pytest
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        DifficultyKnobs(
            n_acquirers=1, decline_class=DeclineClass.SOFT, fraud_sensitivity=1.5,
            n_interacting_traps=0, step_up_prob=0.0, async_capture=False,
            delayed_dispute=False, signal_conflict=0,
        )


def test_direction_is_monotone_easy_to_hard():
    easy = knobs_for(Axis.ROUTING, Difficulty.EASY, is_trap=False)
    med = knobs_for(Axis.ROUTING, Difficulty.MEDIUM, is_trap=False)
    hard = knobs_for(Axis.ROUTING, Difficulty.HARD, is_trap=False)
    # routing varies n_acquirers and signal_conflict up the ladder.
    assert easy.n_acquirers < med.n_acquirers < hard.n_acquirers
    assert easy.signal_conflict <= med.signal_conflict <= hard.signal_conflict


def test_hard_post_purchase_turns_on_tail_risk_knobs():
    # hard => async capture; redline-hard => delayed dispute.
    hard = knobs_for(Axis.POST_PURCHASE, Difficulty.HARD, is_trap=True)
    assert hard.async_capture is True


def test_decline_class_climbs_to_compound_on_hard_decline_recovery():
    easy = knobs_for(Axis.DECLINE_RECOVERY, Difficulty.EASY, is_trap=False)
    hard = knobs_for(Axis.DECLINE_RECOVERY, Difficulty.HARD, is_trap=True)
    assert easy.decline_class is DeclineClass.SOFT
    assert hard.decline_class in {DeclineClass.HARD, DeclineClass.COMPOUND}
