"""Resolve auth/challenge outcomes from a sample's decline_plan (module 10 §3.2).

Two modes, both consuming the same RNG plumbing:
  - scripted: pinned draws (gold/Diamond) — exact determinism for hand-authored ground truth.
  - sampled : seeded Bernoulli/categorical from the DECLINE/CHALLENGE sub-streams.

Determinism (review G2): every draw is keyed by a STABLE per-route authorization-attempt
ordinal (`attempt`), NEVER the mutable `elapsed_steps` clock — so an extra illegal action or
retry never shifts a downstream pinned draw. 3DS challenge resolution lives in its OWN
cleared/failed outcome space (`resolve_challenge_outcome`, keyed on `SubStream.CHALLENGE`); it
is not a re-roll of the auth categorical, which could self-loop the same `challenge` forever.
Pure resolution: returns an outcome; it never mutates the FSM.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from spar.simulator.rng import SubStream, substream

AuthKind = Literal["approve", "decline", "challenge"]
ChallengeKind = Literal["cleared", "failed"]


@dataclass(frozen=True)
class DeclineOutcome:
    """The resolved issuer response for one authorization attempt."""

    kind: AuthKind
    reason_code: str | None = None  # set iff kind == "decline"


@dataclass(frozen=True)
class ChallengeOutcome:
    """The resolved 3DS challenge result (its own outcome space)."""

    kind: ChallengeKind
    reason_code: str | None = None  # set iff kind == "failed"


def _resolve_scripted(plan: dict[str, Any], attempt: int) -> DeclineOutcome:
    for draw in plan.get("draws", []):
        if int(draw["attempt"]) != attempt:
            continue
        outcome = draw["outcome"]
        if outcome == "decline":
            return DeclineOutcome(kind="decline", reason_code=str(draw["reason"]))
        if outcome == "challenge":
            return DeclineOutcome(kind="challenge")
        return DeclineOutcome(kind="approve")
    # No pinned draw at this attempt -> the issuer approves.
    return DeclineOutcome(kind="approve")


def _resolve_sampled(
    plan: dict[str, Any], *, sample_id: str, seed: int, trial_index: int, attempt: int
) -> DeclineOutcome:
    rng = substream(
        sample_id, seed=seed, trial_index=trial_index, stream=SubStream.DECLINE, step=attempt
    )
    p_decline = float(plan.get("p_decline", 0.0))
    p_challenge = float(plan.get("p_challenge", 0.0))
    roll = rng.random()
    if roll < p_challenge:
        return DeclineOutcome(kind="challenge")
    if roll < p_challenge + p_decline:
        soft = list(plan.get("soft_reasons", ["05"]))
        idx = int(rng.integers(0, len(soft)))
        return DeclineOutcome(kind="decline", reason_code=str(soft[idx]))
    return DeclineOutcome(kind="approve")


def resolve_auth_outcome(
    plan: dict[str, Any], *, sample_id: str, seed: int, trial_index: int, attempt: int
) -> DeclineOutcome:
    """Resolve the issuer outcome for authorization attempt `attempt` (a stable ordinal)."""
    if plan.get("mode") == "scripted":
        return _resolve_scripted(plan, attempt)
    return _resolve_sampled(
        plan, sample_id=sample_id, seed=seed, trial_index=trial_index, attempt=attempt
    )


def _resolve_challenge_scripted(plan: dict[str, Any], attempt: int) -> ChallengeOutcome:
    for draw in plan.get("challenge_draws", []):
        if int(draw["attempt"]) != attempt:
            continue
        if draw["outcome"] == "failed":
            return ChallengeOutcome(kind="failed", reason_code=str(draw.get("reason", "05")))
        return ChallengeOutcome(kind="cleared")
    # No pinned challenge draw -> the step-up clears.
    return ChallengeOutcome(kind="cleared")


def _resolve_challenge_sampled(
    plan: dict[str, Any], *, sample_id: str, seed: int, trial_index: int, attempt: int
) -> ChallengeOutcome:
    rng = substream(
        sample_id, seed=seed, trial_index=trial_index, stream=SubStream.CHALLENGE, step=attempt
    )
    p_fail = float(plan.get("p_challenge_fail", 0.0))
    if rng.random() < p_fail:
        reasons = list(plan.get("challenge_fail_reasons", ["05"]))
        idx = int(rng.integers(0, len(reasons)))
        return ChallengeOutcome(kind="failed", reason_code=str(reasons[idx]))
    return ChallengeOutcome(kind="cleared")


def resolve_challenge_outcome(
    plan: dict[str, Any], *, sample_id: str, seed: int, trial_index: int, attempt: int
) -> ChallengeOutcome:
    """Resolve a 3DS challenge for attempt `attempt` in its own cleared/failed space (G2)."""
    if plan.get("mode") == "scripted":
        return _resolve_challenge_scripted(plan, attempt)
    return _resolve_challenge_sampled(
        plan, sample_id=sample_id, seed=seed, trial_index=trial_index, attempt=attempt
    )
