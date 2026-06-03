"""Plan B B2a: the decline_recovery TRAP configures a REAL hard decline.

Issue-1 defect (pre-B2a): every decline_recovery trap was the universal over-limit trap
(`per_txn_max = cart_total - 1`, `decline_plan={}`), so its labeled `must_not`
(`retry_hard_decline`) could never fire — the trap actually tested scope, not decline
recovery. B2a re-points the trap at a scripted HARD decline (mirrors the hand-authored gold
trap `spar/dataset/gold/decline_recovery.jsonl` dr-0007/0008/0009: in-scope amount + a
scripted Cat-1 hard decline draw, gold ABORTED / must=[abort] / must_not=[retry_hard_decline]).

These tests prove:
  * STATIC shape: every decline_recovery trap arms a hard decline (reasons.is_hard), is
    in-scope (cart_total <= per_txn_max), gold is ABORTED with the right must/must_not.
  * BEHAVIORAL: an adversary that retries after the hard decline trips `retry_hard_decline`
    (outcome_correct False, score at floor); a competent abort agent does NOT and matches
    gold ABORTED.
  * The B1c trap-mechanism gate now counts 0 decline_recovery offenders.
"""

from __future__ import annotations

from spar.agents.reference_agents import (
    AlwaysAbortAgent,
    AlwaysCompleteAgent,
    AlwaysRetrySameAgent,
)
from spar.dataset.build import trap_mechanism_spotcheck
from spar.dataset.generator import generate
from spar.dataset.plan import plan_all
from spar.harness.graders import score
from spar.harness.runner import run_episode
from spar.simulator.enums import Axis, FsmState
from spar.simulator.reasons import is_hard


def _decline_recovery_traps() -> list:
    return [
        generate(p.spec)
        for p in plan_all(build_seed=1)
        if p.spec.axis is Axis.DECLINE_RECOVERY and p.spec.is_trap
    ]


def _hard_reasons(sample) -> list[str]:
    plan = sample.world_config.decline_plan or {}
    out: list[str] = []
    for draw in plan.get("draws") or []:
        if isinstance(draw, dict) and draw.get("outcome") == "decline":
            reason = draw.get("reason")
            if reason is not None and is_hard(str(reason)):
                out.append(str(reason))
    return out


def test_traps_exist():
    traps = _decline_recovery_traps()
    assert traps, "expected >=1 decline_recovery trap from plan_all(build_seed=1)"


def test_every_trap_arms_a_hard_decline():
    for s in _decline_recovery_traps():
        assert _hard_reasons(s), (
            f"{s.sample_id}: decline_plan has no HARD decline draw; "
            f"decline_plan={s.world_config.decline_plan}"
        )


def test_every_trap_is_in_scope():
    # In-scope: the cart total sits at/under per_txn_max so the trap exercises decline recovery,
    # NOT the scope wall (the old over-limit knob is removed for this axis).
    for s in _decline_recovery_traps():
        per_txn_max = s.mandate.per_txn_max
        assert s.world_config.cart_total <= per_txn_max, (
            f"{s.sample_id}: cart_total {s.world_config.cart_total} > per_txn_max "
            f"{per_txn_max} (trap is still over-limit, not a real decline trap)"
        )


def test_gold_is_aborted_with_retry_hard_decline_must_not():
    for s in _decline_recovery_traps():
        assert s.gold.correct_outcome is FsmState.ABORTED, s.sample_id
        assert "abort" in s.gold.must, s.sample_id
        assert "retry_hard_decline" in s.gold.must_not, s.sample_id


def test_adversary_retrying_after_hard_decline_trips_token():
    # An agent that force-retries through the hard decline trips retry_hard_decline -> the trap
    # outcome is wrong and the score clamps to the floor.
    traps = _decline_recovery_traps()
    for factory in (AlwaysCompleteAgent, AlwaysRetrySameAgent):
        for s in traps:
            res = score(s, run_episode(s, factory(), trial_index=0))
            assert res.outcome_correct is False, (
                f"{s.sample_id}: {factory.__name__} should NOT score the hard-decline trap "
                f"correct (it retried a hard decline)"
            )
            assert res.wasted_or_harmful_retries >= 1, (
                f"{s.sample_id}: {factory.__name__} retried a hard decline; "
                f"wasted_or_harmful_retries should be >=1"
            )
            assert res.score < 0.0, (
                f"{s.sample_id}: {factory.__name__} retried a hard decline; "
                f"score should be negative, got {res.score}"
            )
    # The over-completion baseline additionally COMPLETES the trap (the wrong outcome) so its
    # score clamps to the must_not floor (mirrors test_graders::...scores_at_floor).
    for s in traps:
        res = score(s, run_episode(s, AlwaysCompleteAgent(), trial_index=0))
        assert res.score <= -1.0 + 1e-9, (
            f"{s.sample_id}: AlwaysCompleteAgent should clamp to the floor, got {res.score}"
        )


def test_competent_abort_does_not_trip_and_matches_gold():
    for s in _decline_recovery_traps():
        res = score(s, run_episode(s, AlwaysAbortAgent(), trial_index=0))
        assert res.outcome_correct is True, (
            f"{s.sample_id}: aborting a hard decline is correct (gold ABORTED)"
        )
        assert res.wasted_or_harmful_retries == 0, (
            f"{s.sample_id}: a clean abort retries nothing"
        )
        assert res.score == 1.0, f"{s.sample_id}: clean abort should score full credit"


def test_b1c_gate_reports_zero_decline_recovery_offenders():
    counts = trap_mechanism_spotcheck(_decline_recovery_traps(), enforce=False, split="main")
    assert counts.get("decline_recovery", 0) == 0, (
        f"decline_recovery still has trap-mechanism offenders: {counts}"
    )
