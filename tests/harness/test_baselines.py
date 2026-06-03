"""Offline baseline scoring: run a reference agent over samples → a build_results summary."""
from spar.agents.reference_agents import AlwaysEscalateAgent, AlwaysAbortAgent
from spar.dataset.loader import load_split
from spar.harness.baselines import score_agent_over_samples


def test_score_agent_returns_summary_with_trust_metrics():
    samples = load_split("lite")  # toy, offline, no network
    res = score_agent_over_samples(AlwaysEscalateAgent, samples)
    s = res["summary"]
    for k in ("trust_score", "trust_score_useful", "any_catastrophic_rate",
              "unsafe_completion_rate", "false_refusal_rate"):
        assert k in s
    assert (s["any_catastrophic_rate"] or 0.0) == 0.0
    assert (s["unsafe_completion_rate"] or 0.0) == 0.0


def test_score_agent_is_deterministic():
    samples = load_split("lite")
    a = score_agent_over_samples(AlwaysAbortAgent, samples)["summary"]
    b = score_agent_over_samples(AlwaysAbortAgent, samples)["summary"]
    assert a == b
