from spar.agents.base import AbortAgent
from spar.harness.runner import run_episode, EpisodeTrace
from spar.simulator.enums import FsmState
from tests.simulator.test_world import _sample  # reuse the fixture builder


def test_run_episode_with_abort_agent_terminates_aborted():
    trace = run_episode(_sample(), AbortAgent(), trial_index=0)
    assert isinstance(trace, EpisodeTrace)
    assert trace.final_state is FsmState.ABORTED
    assert trace.grade_terminal is FsmState.ABORTED  # abort passes through the drain
    assert len(trace.action_log) == 1


def test_run_episode_happy_path_grade_terminal_is_closed():
    from spar.agents.reference_agents import HappyPathAgent
    trace = run_episode(_sample(), HappyPathAgent(), trial_index=0)
    assert trace.final_state is FsmState.SETTLED       # agent-terminal
    assert trace.grade_terminal is FsmState.CLOSED     # grade-terminal after drain (F3)


def test_run_episode_respects_step_budget():
    # An agent that never terminates should hit the cap and be force-aborted.
    class Spinner:
        def act(self, obs):
            from spar.simulator.contract import ComputeTax
            return ComputeTax(tool="compute_tax")
    sample = _sample()
    sample.world_config.max_steps = 3
    trace = run_episode(sample, Spinner(), trial_index=0)
    assert trace.final_state is FsmState.ABORTED
    assert trace.abort_reason == "step_budget_exhausted"  # F6: must not credit gold ABORTED
