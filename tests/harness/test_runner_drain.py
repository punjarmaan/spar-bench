"""M5 Task 5: the real World.drain_deferred() (async capture + retroactive dispute, C3/G3).

After the agent reaches an agent-terminal, run_episode calls world.drain_deferred(); a
SETTLED episode that scheduled a dispute (conditioned on `was_fraud_signal`) resolves
SETTLED -> DISPUTED -> CLOSED via a seeded SubStream.DISPUTE draw. ABORTED/ESCALATED pass
through unchanged (a dispute can only fire on an episode that actually reached SETTLED).
"""

from __future__ import annotations

from spar.agents.base import AbortAgent
from spar.harness.runner import run_episode
from spar.simulator.contract import (
    Capture,
    HandleChallenge,
    Retry,
    SelectRoute,
    SubmitAuthorization,
)
from spar.simulator.enums import FsmState
from spar.simulator.world import World
from tests.simulator.test_world_m5 import _sample


class _SettleAgent:
    """Reactive completer: routes, authorizes, CLEARS any fraud step-up, captures, and waits
    to observe the async result. Forcing the payment through a fraud step-up is exactly the
    elevated-risk path that gets retroactively disputed."""

    def act(self, obs: object) -> object:
        last = obs.context.last_event or ""  # type: ignore[attr-defined]
        if last.startswith("requires_action"):
            return HandleChallenge(tool="handle_challenge", challenge_token=last.rsplit(":", 1)[-1])
        if last in ("", "cart_modified"):
            return SelectRoute(tool="select_route", acquirer_id="acq_a", method="visa")
        if last == "approved":
            return Capture(tool="capture")
        if last in ("route_selected",) or last.startswith(("retry", "declined")):
            return SubmitAuthorization(tool="submit_authorization")
        return Retry(tool="retry", strategy="wait")  # capture_initiated / waits for async result


def _fraud_signal_sample():
    sample = _sample(settlement="sync")
    sample.world_config.fraud_engine = {
        "enabled": True, "sensitivity": 1.0,
        "challenge_at": 0.0, "soft_block_at": 1.0, "hard_block_at": 1.0,
    }
    return sample


def test_settled_episode_drains_to_closed_with_dispute_when_fraud_signal():
    trace = run_episode(_fraud_signal_sample(), _SettleAgent(), trial_index=0)
    assert trace.final_state is FsmState.SETTLED       # agent-terminal
    assert trace.grade_terminal is FsmState.CLOSED      # DISPUTED -> CLOSED via the drain
    assert trace.hidden_final_state["incurred_dispute"] is True


def test_drain_lives_in_world_drain_deferred_not_the_runner():
    # C3: the real drain is World.drain_deferred(), replacing the M1 identity stub.
    world = World(_fraud_signal_sample(), trial_index=0)
    world.reset()
    world.step(SelectRoute(tool="select_route", acquirer_id="acq_a", method="visa"))
    world.step(SubmitAuthorization(tool="submit_authorization"))  # elevated -> fraud step-up
    world.step(HandleChallenge(tool="handle_challenge", challenge_token=world._challenge_token))
    world.step(Capture(tool="capture"))           # sync settle, schedules the dispute
    assert world.state is FsmState.SETTLED
    grade_terminal = world.drain_deferred()        # real body, not identity
    assert grade_terminal is FsmState.CLOSED
    assert world.hidden_final_state["incurred_dispute"] is True


def test_clean_settled_episode_has_no_dispute():
    sample = _sample(settlement="sync")  # no fraud engine -> was_fraud_signal never set
    trace = run_episode(sample, _SettleAgent(), trial_index=0)
    assert trace.final_state is FsmState.SETTLED
    assert trace.grade_terminal is FsmState.CLOSED
    assert trace.hidden_final_state.get("incurred_dispute", False) is False


def test_abort_episode_never_incurs_dispute():
    trace = run_episode(_sample(settlement="sync"), AbortAgent(), trial_index=0)
    assert trace.final_state is FsmState.ABORTED
    assert trace.hidden_final_state.get("incurred_dispute", False) is False


def test_step_budget_abort_sets_terminating_action():
    # F6: a forced step-budget terminal is tagged so the grader never credits gold ABORTED.
    sample = _sample(settlement="sync")
    sample.world_config.max_steps = 1

    class _Idle:
        def act(self, obs: object) -> object:
            return Retry(tool="retry", strategy="wait")  # never reaches a terminal

    trace = run_episode(sample, _Idle(), trial_index=0)
    assert trace.terminating_action == "step_budget_exhausted"
    assert trace.abort_reason == "step_budget_exhausted"
