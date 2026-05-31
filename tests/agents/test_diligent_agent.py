from spar.agents.reference_agents import DiligentAgent, AlwaysCompleteAgent
from spar.harness.runner import run_episode
from spar.harness.graders import score
from spar.dataset.applicability import applicability_from_world_config
from spar.dataset.generator import generate, GenSpec
from spar.simulator.enums import Axis, Difficulty


def _trap(axis, seed=1):
    # Build-time applicability is what gold.expected_violations carries in a real dataset
    # (build._stamp_applicability); generate() alone leaves it empty, so stamp it here so the
    # catastrophic gate is actually exercised (mirrors the foundation-baseline test fixtures).
    sample = generate(GenSpec(axis=axis, seed=seed, difficulty=Difficulty.EASY, is_trap=True))
    sample.gold.expected_violations = sorted(
        applicability_from_world_config(sample), key=lambda v: v.value
    )
    assert sample.gold.expected_violations, "seed produced no catastrophic trap"
    return sample


def test_diligent_agent_avoids_catastrophe_where_always_complete_trips():
    # consent_mandate trap with seed=1 maps to the mandate_breach class — the gate is
    # meaningful only because expected_violations is non-empty (asserted in _trap).
    sample = _trap(Axis.CONSENT_MANDATE)
    dil = score(sample, run_episode(sample, DiligentAgent(), trial_index=0))
    com = score(sample, run_episode(sample, AlwaysCompleteAgent(), trial_index=0))
    assert dil.catastrophic_class is None          # diligent avoided it
    assert com.catastrophic_class is not None       # always-complete tripped it
