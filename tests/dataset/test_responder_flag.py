from spar.dataset.generator import generate, GenSpec
from spar.simulator.enums import Axis, Difficulty, IntentSpec

def test_nontrap_gets_responder_true_trap_false():
    nontrap = generate(GenSpec(axis=Axis.COMPLIANCE_TAX, seed=3, difficulty=Difficulty.EASY,
                               is_trap=False, intent_spec=IntentSpec.EXPLICIT))
    trap = generate(GenSpec(axis=Axis.COMPLIANCE_TAX, seed=3, difficulty=Difficulty.EASY,
                            is_trap=True, intent_spec=IntentSpec.EXPLICIT))
    assert nontrap.world_config.responder is True
    assert trap.world_config.responder is False

def test_responder_true_across_axes_for_nontraps():
    for axis in Axis:
        s = generate(GenSpec(axis=axis, seed=5, difficulty=Difficulty.MEDIUM,
                             is_trap=False, intent_spec=IntentSpec.EXPLICIT))
        assert s.world_config.responder is True, axis
