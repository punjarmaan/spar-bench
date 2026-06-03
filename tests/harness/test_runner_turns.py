"""run_episode threads the agent's per-turn records onto EpisodeTrace (enrichment T4)."""
from spar.eval.agent import ModelAgent
from spar.eval.profile import StageSampling
from spar.harness.runner import run_episode
from spar.dataset.loader import load_split


def _fn(content):
    def _f(**kwargs):
        msg = type("M", (), {"content": content, "reasoning_content": "because"})()
        choice = type("C", (), {"message": msg})()
        return type("R", (), {
            "choices": [choice],
            "usage": type("U", (), {"prompt_tokens": 50, "completion_tokens": 4})(),
            "_hidden_params": {"response_cost": 0.0},
        })()
    return _f


def test_trace_carries_agent_turns():
    sample = load_split("lite")[0]
    agent = ModelAgent(
        route="r", policy_text="p", sampling=StageSampling(temperature=0.0),
        supports_response_format=False,
        completion_fn=_fn('{"tool": "abort", "args": {"reason": "x"}}'),
        mandate_text="m",
    )
    trace = run_episode(sample, agent)
    assert hasattr(trace, "agent_turns")
    assert len(trace.agent_turns) >= 1
    assert trace.agent_turns[0].reasoning == "because"


def test_non_model_agent_yields_empty_turns():
    class _Dummy:
        def act(self, obs):
            from spar.simulator.contract import Abort
            return Abort(tool="abort", reason="x")
    sample = load_split("lite")[0]
    trace = run_episode(sample, _Dummy())
    assert trace.agent_turns == []  # defensive getattr default
