from spar.harness.graders import SampleScore, score
from spar.harness.runner import run_episode
from spar.agents.base import AbortAgent
from tests.simulator.test_world import _sample


def test_stub_score_marks_abort_against_closed_gold_as_incorrect():
    sample = _sample()  # gold.correct_outcome == CLOSED
    trace = run_episode(sample, AbortAgent(), trial_index=0)
    s = score(sample, trace)
    assert isinstance(s, SampleScore)
    assert s.outcome_correct is False
    assert s.score == 0.0


def test_stub_score_credits_happy_path_via_grade_terminal():
    from spar.agents.reference_agents import HappyPathAgent
    sample = _sample()  # gold.correct_outcome == CLOSED
    trace = run_episode(sample, HappyPathAgent(), trial_index=0)
    s = score(sample, trace)
    # SETTLED agent-terminal drains to CLOSED grade-terminal == gold → credited.
    assert s.outcome_correct is True and s.score == 1.0
