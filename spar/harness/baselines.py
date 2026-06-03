"""Offline reference-agent scoring (build gate). Drives a scripted Agent through the real
run_episode + score + build_results — no model calls — to compute a baseline's summary. Used to
quantify the always-escalate gaming floor and as a build-time honesty gate.

Mirrors the orchestrator's offline loop exactly: a deny ScriptedUserSim terminates every
escalation, a StubModelGrader supplies the Tier-C surface with no network, score is called with
`model_grader=`, and build_results echoes `weights=DEFAULT_WEIGHTS.as_dict()` (the same dict form
the orchestrator passes). Deterministic."""

from __future__ import annotations

from typing import Any, Callable

from spar.harness.graders import score
from spar.harness.model_grader import StubModelGrader
from spar.harness.report import build_results
from spar.harness.runner import run_episode
from spar.harness.user_sim import ScriptedUserSim, UserResponse
from spar.harness.weights import DEFAULT_WEIGHTS
from spar.simulator.schemas import Sample


def score_agent_over_samples(
    agent_factory: Callable[[], Any],
    samples: list[Sample],
    *,
    canary: str = "spar:baseline",
    build_seed: int = 0,
) -> dict[str, Any]:
    """Score a zero-arg agent class/factory over `samples` offline. Returns a build_results dict
    (with a `summary` block carrying trust_score, *_useful, any_catastrophic_rate,
    unsafe_completion_rate, false_refusal_rate). No model calls: the responder always denies (so
    escalations terminate) and the grader is the deterministic offline stub. agent_factory is
    called once per sample so each episode gets a fresh, stateful agent instance."""
    grader = StubModelGrader()
    # Deny responder: every escalation terminates with a clamped-to-zero bound (never launders an
    # approval). Matches the orchestrator's offline wiring.
    responder = ScriptedUserSim(UserResponse(decision="deny"))
    scores = [
        score(
            sample,
            run_episode(sample, agent_factory(), trial_index=0, user_sim=responder),
            model_grader=grader,
        )
        for sample in samples
    ]
    return build_results(
        scores,
        split="baseline",
        canary=canary,
        build_seed=build_seed,
        weights=DEFAULT_WEIGHTS.as_dict(),
        enforce_cap=False,
    )
