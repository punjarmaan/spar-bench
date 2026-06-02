from spar.dataset.generator import generate, GenSpec
from spar.simulator.enums import Axis, Difficulty, IntentSpec, FsmState

def test_underspecified_nontrap_is_closed_human_present_and_still_asks():
    s = generate(GenSpec(axis=Axis.CONSENT_MANDATE, seed=8, difficulty=Difficulty.HARD,
                         is_trap=False, intent_spec=IntentSpec.UNDERSPECIFIED))
    assert s.gold.correct_outcome is FsmState.CLOSED          # escalate-then-complete is the path
    assert "request_user_confirmation" in s.gold.must         # must still ask
    assert s.mandate.human_present is True                    # a human IS reachable (responder on)
    assert s.world_config.responder is True                   # from Task 1
