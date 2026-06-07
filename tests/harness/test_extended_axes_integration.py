"""End-to-end axis grading: run reference agents through the gold and grade via score().

Proves the exit-criteria behaviors: deterministic deferred draining per seed, async-fail
detection (a never-verify agent scores poorly), and the retroactive dispute correlating
with `was_fraud_signal`. Grading goes through the unified `score()` — there is no
separate axis grader.
"""

from __future__ import annotations

from importlib import resources

from spar.harness.graders import score
from spar.harness.runner import run_episode
from spar.harness.weights import DEFAULT_WEIGHTS
from spar.simulator.contract import (
    Abort,
    Capture,
    ComputeTax,
    HandleChallenge,
    Retry,
    SelectRoute,
    SubmitAuthorization,
)
from spar.simulator.enums import FsmState
from spar.simulator.schemas import Sample


def _load(axis: str) -> list[Sample]:
    text = resources.files("spar.dataset.gold").joinpath(f"{axis}.jsonl").read_text("utf-8")
    return [Sample.model_validate_json(line) for line in text.splitlines() if line.strip()]


class _VerifyAgent:
    """Computes tax, routes, authorizes, CLEARS any step-up, captures, and waits to observe
    the async capture result before considering the purchase done."""

    def __init__(self) -> None:
        self._stage = 0

    def act(self, obs: object) -> object:
        last = obs.context.last_event or ""  # type: ignore[attr-defined]
        if last.startswith("requires_action"):
            return HandleChallenge(tool="handle_challenge", challenge_token=last.rsplit(":", 1)[-1])
        if last == "capture_initiated":
            return Retry(tool="retry", strategy="wait")  # observe the deferred result
        self._stage += 1
        if self._stage == 1:
            return ComputeTax(tool="compute_tax")
        if self._stage == 2:
            return SelectRoute(tool="select_route", acquirer_id="acq_a", method="visa")
        if self._stage == 3:
            return SubmitAuthorization(tool="submit_authorization")
        if self._stage == 4:
            return Capture(tool="capture")
        return Retry(tool="retry", strategy="wait")


class _NeverVerifyAgent:
    """Captures (async) then declares done via Abort BEFORE the slow result fires (trap
    latency=3), so it never observes the outcome -> verify_capture_result is unsatisfied."""

    def __init__(self) -> None:
        self._stage = 0

    def act(self, obs: object) -> object:
        self._stage += 1
        if self._stage == 1:
            return ComputeTax(tool="compute_tax")
        if self._stage == 2:
            return SelectRoute(tool="select_route", acquirer_id="acq_a", method="visa")
        if self._stage == 3:
            return SubmitAuthorization(tool="submit_authorization")
        if self._stage == 4:
            return Capture(tool="capture")
        return Abort(tool="abort", reason="assumed-success")


def test_post_purchase_legit_verify_agent_scores_full_credit():
    legit = [s for s in _load("post_purchase") if not s.is_trap][0]
    trace = run_episode(legit, _VerifyAgent(), trial_index=0)
    assert trace.final_state is FsmState.SETTLED
    s = score(legit, trace, weights=DEFAULT_WEIGHTS)
    assert s.outcome_correct is True and s.score == 1.0


def test_never_verify_agent_scores_poorly_on_async_fail():
    trap = [s for s in _load("post_purchase") if s.is_trap][0]  # capture_fail=true, latency=3
    trace = run_episode(trap, _NeverVerifyAgent(), trial_index=0)
    s = score(trap, trace, weights=DEFAULT_WEIGHTS)
    assert s.outcome_correct is False
    assert s.score < 1.0


def test_deferred_drain_is_deterministic_per_seed():
    legit = [s for s in _load("post_purchase") if not s.is_trap][0]
    a = run_episode(legit, _VerifyAgent(), trial_index=0)
    b = run_episode(legit, _VerifyAgent(), trial_index=0)
    assert a.grade_terminal == b.grade_terminal
    assert a.hidden_final_state == b.hidden_final_state


def test_retroactive_dispute_correlates_with_fraud_signal():
    # Inject a fraud signal + a certain dispute: the cleared-step-up-then-pay path latches
    # was_fraud_signal, so the post-settle dispute fires (prob_fraud=1.0) and sinks the score.
    legit = [s for s in _load("post_purchase") if not s.is_trap][0]
    legit.world_config.dispute = {"fires_at_offset": 2, "prob_fraud": 1.0, "prob_clean": 0.0}
    legit.world_config.fraud_engine = {
        "enabled": True, "sensitivity": 1.0,
        "challenge_at": 0.0, "soft_block_at": 1.0, "hard_block_at": 1.0,
    }
    trace = run_episode(legit, _VerifyAgent(), trial_index=0)
    s = score(legit, trace, weights=DEFAULT_WEIGHTS)
    assert s.incurred_dispute is True
    assert s.score < 1.0
