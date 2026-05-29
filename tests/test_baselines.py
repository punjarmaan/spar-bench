from spar.agents.reference_agents import AlwaysCompleteAgent
from spar.dataset.loader import load_gold
from spar.harness.graders import score
from spar.harness.report import build_results
from spar.harness.runner import run_episode
from spar.harness.weights import DEFAULT_WEIGHTS
from spar.simulator.enums import Axis


def _consent_traps():
    return [s for s in load_gold(Axis.CONSENT_MANDATE) if s.is_trap]


def test_always_complete_each_trap_at_or_below_floor_on_trap_subset():
    # V1: each consent trap's per-sample score <= score_floor, asserted DIRECTLY.
    floor = DEFAULT_WEIGHTS.score_floor
    for sample in _consent_traps():
        trace = run_episode(sample, AlwaysCompleteAgent(), trial_index=0)
        s = score(sample, trace)
        assert s.score <= floor, f"{sample.sample_id} scored {s.score} > floor {floor}"
        assert s.unsafe_completion is True, sample.sample_id


def test_always_complete_competence_present_but_gated_to_zero():
    # V1: competence demonstrably present (axis_mean > 0 on non-traps) AND trust_score gated
    # to ~0 over the full gold — distinguishes "the gate works" from "the agent is just bad".
    full, non_traps = [], []
    for sample in load_gold(Axis.CONSENT_MANDATE):
        trace = run_episode(sample, AlwaysCompleteAgent(), trial_index=0)
        try:
            s = score(sample, trace)
        except NotImplementedError:
            continue  # Tier C is M6; not part of the M4 gate proof
        full.append(s)
        if not sample.is_trap:
            non_traps.append(s)

    competence = build_results(non_traps, split="lite", canary="spar:t", build_seed=0,
                               weights=DEFAULT_WEIGHTS.as_dict())
    assert competence["summary"]["axis_mean"] > 0.0  # competence demonstrably present

    results = build_results(full, split="lite", canary="spar:t", build_seed=0,
                            weights=DEFAULT_WEIGHTS.as_dict())
    assert results["summary"]["overspend_rate"] >= 0.95  # nearly every trap completed
    assert results["summary"]["trust_score"] == 0.0       # gate collapses it (F1)
