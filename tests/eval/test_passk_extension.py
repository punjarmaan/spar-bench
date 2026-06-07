"""run_trials threads the pinned responder + grader (backwards compatible)."""

from __future__ import annotations

from spar.agents.base import Agent
from spar.dataset.loader import load_split
from spar.harness import passk as passk_mod
from spar.harness.passk import TrialResult, run_trials
from spar.simulator.contract import Abort, Action, Observation


class _AbortAgent:
    """Always aborts — a trivial importable Agent for offline trials."""

    def act(self, observation: Observation) -> Action:
        return Abort(tool="abort", reason="test")


def _factory() -> Agent:
    return _AbortAgent()


def test_run_trials_old_signature_still_works() -> None:
    sample = load_split("lite")[0]
    result = run_trials(sample, _factory, k=1)
    assert isinstance(result, TrialResult)
    assert result.n == 1
    assert 0 <= result.c <= 1


def test_run_trials_threads_user_sim_and_grader(monkeypatch) -> None:
    sample = load_split("lite")[0]
    seen: dict[str, object] = {}

    real_run_episode = passk_mod.run_episode
    real_score = passk_mod.score

    def spy_run_episode(s, a, *, trial_index=0, user_sim=None):
        seen["user_sim"] = user_sim
        return real_run_episode(s, a, trial_index=trial_index, user_sim=user_sim)

    def spy_score(s, trace, *, model_grader=None, **kw):
        seen["model_grader"] = model_grader
        return real_score(s, trace, model_grader=model_grader, **kw)

    monkeypatch.setattr(passk_mod, "run_episode", spy_run_episode)
    monkeypatch.setattr(passk_mod, "score", spy_score)

    sentinel_sim = object()
    sentinel_grader = object()
    result = run_trials(
        sample, _factory, k=2, user_sim=sentinel_sim, model_grader=sentinel_grader
    )
    assert isinstance(result, TrialResult)
    assert result.n == 2
    assert seen["user_sim"] is sentinel_sim
    assert seen["model_grader"] is sentinel_grader
