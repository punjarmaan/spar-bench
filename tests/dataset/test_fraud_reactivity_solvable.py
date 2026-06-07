from spar.dataset.generator import generate, GenSpec
from spar.simulator.enums import Axis, Difficulty, IntentSpec
from spar.agents.reference_agents import AlwaysCompleteAgent
from spar.agents.naive_complete import NaiveCompleteAgent
from spar.harness.runner import run_episode
from spar.harness.graders import score

CHALLENGE_AXES = (Axis.FRAUD_REACTIVITY, Axis.DECLINE_RECOVERY)

def _easy_nontrap(axis, seed):
    return generate(GenSpec(axis=axis, seed=seed, difficulty=Difficulty.EASY,
                            is_trap=False, intent_spec=IntentSpec.EXPLICIT))

def test_easy_challenge_axes_fire_a_challenge():
    # AlwaysCompleteAgent submits -> a CHALLENGE must fire (latches challenge_step) for BOTH axes.
    for axis in CHALLENGE_AXES:
        for seed in (1, 2, 3, 7, 11):
            s = _easy_nontrap(axis, seed)
            trace = run_episode(s, AlwaysCompleteAgent(), trial_index=0)
            assert trace.hidden_final_state.get("challenge_step") is not None, (axis, seed)

def test_easy_challenge_axes_still_fail_naive():
    # A naive agent that ignores the challenge must NOT complete correctly.
    for axis in CHALLENGE_AXES:
        for seed in (1, 2, 3, 7, 11):
            s = _easy_nontrap(axis, seed)
            assert score(s, run_episode(s, NaiveCompleteAgent(), trial_index=0)).outcome_correct is False
