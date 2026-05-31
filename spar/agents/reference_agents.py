"""Reference baseline agents (module 20 §5).

Each satisfies the frozen spar.agents.base.Agent protocol. Seeded agents draw only
through spar.simulator.rng. These are frozen for downstream milestones (M3+ import them
as fixed baselines for the validation gate).
"""

from __future__ import annotations

from typing import Any

from spar.simulator.contract import (
    Abort,
    Action,
    Capture,
    HandleChallenge,
    Observation,
    Refund,
    Retry,
    SelectRoute,
    SubmitAuthorization,
    parse_action,
)
from spar.simulator.rng import SubStream, substream


class HappyPathAgent:
    """Selects the first route, authorizes, captures. For M1 smoke tests only."""

    def __init__(self) -> None:
        self._step = 0

    def act(self, observation: Observation) -> Action:
        self._step += 1
        if self._step == 1:
            acq = observation.methods[0]
            return SelectRoute(tool="select_route", acquirer_id=acq.acquirer_id, method=acq.methods[0])
        if self._step == 2:
            return SubmitAuthorization(tool="submit_authorization")
        return Capture(tool="capture")


def _last_event(obs: Observation) -> str | None:
    ev = obs.context.recent_events
    return ev[-1] if ev else None


def _challenge_pending(obs: Observation) -> bool:
    last = _last_event(obs)
    return last is not None and last.startswith("requires_action:1A")


def _live_challenge_token(obs: Observation) -> str:
    """Read the LIVE challenge token from the surfaced event (never recompute it)."""
    last = _last_event(obs) or ""
    parts = last.split(":")
    return parts[2] if len(parts) >= 3 else "chal_unknown"


def _was_declined(obs: Observation) -> bool:
    return obs.context.last_reason_code is not None and not _challenge_pending(obs)


class AlwaysAbortAgent:
    """Always aborts on the first turn. The over-refusal baseline."""

    def act(self, observation: Observation) -> Action:
        return Abort(tool="abort", reason="always-abort baseline")


class AlwaysCompleteAgent:
    """Drives route -> auth -> capture and force-retries through any decline.

    The over-completion baseline: it completes traps (hard declines) by hammering,
    so it must score low on the hard-decline trap subset (the M2 exit criterion).
    """

    def __init__(self) -> None:
        self._selected = False
        self._submitted = False

    def act(self, observation: Observation) -> Action:
        if _challenge_pending(observation):
            return HandleChallenge(
                tool="handle_challenge", challenge_token=_live_challenge_token(observation)
            )
        if _was_declined(observation):
            # A genuine decline -> retry the same route and re-authorize (routing/decline path).
            self._submitted = False
            return Retry(tool="retry", strategy="same")
        if not self._selected and observation.methods:
            self._selected = True
            m = observation.methods[0]
            return SelectRoute(
                tool="select_route", acquirer_id=m.acquirer_id, method=m.methods[0]
            )
        if not self._submitted:
            self._submitted = True
            return SubmitAuthorization(tool="submit_authorization")
        # Force the capture even after a silent scope-violation submit (which records no
        # decline): this drives the consent scope wall, flipping World.unsafe_completion (F1).
        return Capture(tool="capture")


class LateRefundAgent:
    """Focused adversary that trips `missed_reversal_window` (Task 4.3b / C8).

    AlwaysCompleteAgent cannot trip this class: it stops at capture, and a refund within a
    zero-length window is still legal at post-capture ordinal 0. This agent drives
    select -> submit_authorization -> capture, then takes ONE extra post-capture MUTATING action
    (a second capture — illegal, but it advances the world's post-capture refund-window ordinal),
    and only THEN issues a `Refund`. With the trap's `refund_window_steps=0`, a refund past the
    window latches `missed_reversal_window` in hidden_final_state. It targets the async
    missed_reversal trap (which stays CAPTURE_PENDING — non-terminal — so the post-capture action
    and refund both execute before any agent-terminal). After the late refund it aborts to end
    the episode deterministically. Minimal, deterministic, no RNG.
    """

    def __init__(self) -> None:
        self._selected = False
        self._submitted = False
        self._captured = False
        self._bumped = False
        self._refunded = False

    def act(self, observation: Observation) -> Action:
        if _challenge_pending(observation):
            return HandleChallenge(
                tool="handle_challenge", challenge_token=_live_challenge_token(observation)
            )
        if not self._captured and _was_declined(observation):
            # Force through a decline so the agent reliably reaches APPROVED -> capture (the
            # late-refund mechanic needs a committed capture regardless of difficulty rung).
            self._submitted = False
            return Retry(tool="retry", strategy="same")
        if not self._selected and observation.methods:
            self._selected = True
            m = observation.methods[0]
            return SelectRoute(
                tool="select_route", acquirer_id=m.acquirer_id, method=m.methods[0]
            )
        if not self._submitted:
            self._submitted = True
            return SubmitAuthorization(tool="submit_authorization")
        if not self._captured:
            self._captured = True
            return Capture(tool="capture")
        if not self._bumped:
            # Advance the post-capture refund-window ordinal past the (zero-length) window with a
            # mutating action, so the subsequent refund is provably LATE (world counts this step).
            self._bumped = True
            return Capture(tool="capture")
        if not self._refunded:
            self._refunded = True
            return Refund(tool="refund")
        return Abort(tool="abort", reason="late-refund adversary done")


class CombinedViolationAgent:
    """Deterministic select -> submit_authorization -> capture driver for combo traps (C19).

    On a multi-knob catastrophic trap whose scope-CLEAN issuer_behavior carries several
    divergence knobs at once (e.g. `settled_payee` diverged + `settled_currency` diverged +
    `auth_only=True`), a single committed capture latches SEVERAL catastrophic flags in ONE
    trajectory: misdirected_funds + wrong_currency (via `_record_capture_divergence_flags`)
    and premature_capture (auth_only warranted). This is the multi-class baseline that proves
    the worst-class collapse (one tag, never a multiset) on a trajectory with >= 2 unreduced
    flags. Minimal, deterministic, no RNG — it never refuses and force-retries through declines
    so it reliably reaches APPROVED -> capture regardless of difficulty rung.
    """

    def __init__(self) -> None:
        self._selected = False
        self._submitted = False

    def act(self, observation: Observation) -> Action:
        if _challenge_pending(observation):
            return HandleChallenge(
                tool="handle_challenge", challenge_token=_live_challenge_token(observation)
            )
        if _was_declined(observation):
            self._submitted = False
            return Retry(tool="retry", strategy="same")
        if not self._selected and observation.methods:
            self._selected = True
            m = observation.methods[0]
            return SelectRoute(
                tool="select_route", acquirer_id=m.acquirer_id, method=m.methods[0]
            )
        if not self._submitted:
            self._submitted = True
            return SubmitAuthorization(tool="submit_authorization")
        return Capture(tool="capture")


class AlwaysRetrySameAgent:
    """Selects a route, authorizes, and retries `same` on every decline forever."""

    def __init__(self) -> None:
        self._started = False

    def act(self, observation: Observation) -> Action:
        if _was_declined(observation):
            return Retry(tool="retry", strategy="same")
        if not self._started:
            self._started = True
            if observation.methods:
                m = observation.methods[0]
                return SelectRoute(
                    tool="select_route", acquirer_id=m.acquirer_id, method=m.methods[0]
                )
        return SubmitAuthorization(tool="submit_authorization")


class RandomAgent:
    """Seeded random action chooser (over a safe legal-shaped subset). Reproducible."""

    _TOOLS = ("select_route", "submit_authorization", "capture", "retry", "abort")

    def __init__(self, *, seed: int) -> None:
        self.seed = seed
        self._step = 0

    def act(self, observation: Observation) -> Action:
        rng = substream(
            "random_agent", seed=self.seed, trial_index=0,
            stream=SubStream.DECLINE, step=self._step,
        )
        self._step += 1
        choice = self._TOOLS[int(rng.integers(0, len(self._TOOLS)))]
        if choice == "select_route" and observation.methods:
            m = observation.methods[0]
            return SelectRoute(
                tool="select_route", acquirer_id=m.acquirer_id, method=m.methods[0]
            )
        if choice == "submit_authorization":
            return SubmitAuthorization(tool="submit_authorization")
        if choice == "capture":
            return Capture(tool="capture")
        if choice == "retry":
            return Retry(tool="retry", strategy="same")
        return Abort(tool="abort", reason="random baseline")


class GoldReplayAgent:
    """Replays a gold `trajectory` (a list of action dicts) VERBATIM.

    Per the review + ownership registry: it plays `gold.trajectory` step-by-step via
    `parse_action`, in order, one action per `act` call. It does NOT recompute the hidden
    `challenge_token` from a step counter (that desyncs). When the next verbatim step is a
    `handle_challenge`, it substitutes the LIVE token surfaced in the observation event
    (`requires_action:1A:<token>`). When the trajectory is exhausted it aborts (a defensive
    terminal; a well-formed gold trajectory ends on its own terminal action).

    Build it from a sample: `GoldReplayAgent(trajectory=sample.gold.trajectory)`.
    """

    def __init__(self, *, trajectory: list[dict[str, Any]]) -> None:
        self._trajectory = [dict(step) for step in trajectory]
        self._i = 0

    def act(self, observation: Observation) -> Action:
        if self._i >= len(self._trajectory):
            return Abort(tool="abort", reason="gold trajectory exhausted")
        step = dict(self._trajectory[self._i])
        self._i += 1
        if step.get("tool") == "handle_challenge":
            # Substitute the live token from the observation; never recompute it.
            step["challenge_token"] = _live_challenge_token(observation)
        return parse_action(step)
