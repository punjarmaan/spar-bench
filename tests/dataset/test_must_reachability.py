import pytest
from spar.dataset.generator import generate, GenSpec
from spar.simulator.enums import Axis, Difficulty, IntentSpec
from spar.dataset.build import must_reachability_spotcheck

def _easy_fr_nontrap():
    return generate(GenSpec(axis=Axis.FRAUD_REACTIVITY, seed=2, difficulty=Difficulty.EASY,
                            is_trap=False, intent_spec=IntentSpec.EXPLICIT))

def test_gate_passes_on_fixed_fraud_nontrap():
    must_reachability_spotcheck([_easy_fr_nontrap()])   # no raise

def test_gate_raises_on_unreachable_handle_challenge():
    # Mutate a good sample to disable its fraud engine -> handle_challenge becomes unreachable.
    s = _easy_fr_nontrap()
    broken = s.model_copy(update={
        "world_config": s.world_config.model_copy(update={"fraud_engine": {"enabled": False}})
    })
    with pytest.raises(AssertionError, match="handle_challenge"):
        must_reachability_spotcheck([broken])

def test_gate_ignores_traps():
    trap = generate(GenSpec(axis=Axis.FRAUD_REACTIVITY, seed=2, difficulty=Difficulty.EASY,
                            is_trap=True, intent_spec=IntentSpec.EXPLICIT))
    must_reachability_spotcheck([trap])   # traps are not checked; no raise

def test_gate_passes_when_decline_plan_provides_challenge_source():
    # fraud engine off, but a decline_plan challenge mode is a valid challenge source -> reachable.
    s = _easy_fr_nontrap()
    ok = s.model_copy(update={
        "world_config": s.world_config.model_copy(update={
            "fraud_engine": {"enabled": False},
            "decline_plan": {"mode": "challenge"},
        })
    })
    must_reachability_spotcheck([ok])   # no raise
