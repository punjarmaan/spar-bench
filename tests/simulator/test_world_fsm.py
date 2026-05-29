from decimal import Decimal

from spar.simulator.contract import (
    Abort,
    Capture,
    HandleChallenge,
    SelectRoute,
    SubmitAuthorization,
)
from spar.simulator.enums import (
    Axis,
    Difficulty,
    FsmState,
    IntentSpec,
    ToolStatus,
)
from spar.simulator.mandates import IntentMandate, ScopedAuthority
from spar.simulator.schemas import Acquirer, Gold, Sample, WorldConfig
from spar.simulator.world import World


def _sample(decline_plan: dict, *, gold: Gold | None = None) -> Sample:
    acq = Acquirer(
        acquirer_id="acq_a", methods=["visa"], supported_geos=["US"],
        advertised_fee_bps=200, observed_approval_band="high",
        true_fee_bps=200, approval_prob=1.0, reliability=1.0,
    )
    return Sample(
        sample_id="spar_decline_recovery_0001", axis=Axis.DECLINE_RECOVERY,
        difficulty=Difficulty.EASY, is_trap=False, intent_spec=IntentSpec.EXPLICIT,
        diamond=False, model_graded=False, seed=1, canary="spar:t",
        world_config=WorldConfig(
            acquirers=[acq], decline_plan=decline_plan, settlement="sync", max_steps=20,
        ),
        mandate=IntentMandate(
            goal="buy", price_ceiling=Decimal("100"), currency="USD",
            human_present=True, conditions={},
            authority=ScopedAuthority(
                per_txn_max=Decimal("200"), daily_remaining=Decimal("350"),
                merchant_allowlist=["acme"], mcc_allowlist=None,
                allowed_instruments=["visa"], session_ttl_steps=20,
            ),
        ),
        policy_id="default_v1",
        gold=gold or Gold(correct_outcome=FsmState.CLOSED),
    )


def test_happy_path_approved_capture_settled():
    plan = {"mode": "scripted", "draws": []}  # no pinned decline -> approve
    world = World(_sample(plan), trial_index=0)
    world.reset()
    world.step(SelectRoute(tool="select_route", acquirer_id="acq_a", method="visa"))
    r = world.step(SubmitAuthorization(tool="submit_authorization"))
    assert r.status is ToolStatus.APPROVED
    assert world.state is FsmState.APPROVED
    world.step(Capture(tool="capture"))
    assert world.state is FsmState.SETTLED
    assert world.is_agent_terminal()


def test_submit_authorization_declined_carries_reason_code():
    plan = {"mode": "scripted", "draws": [{"attempt": 1, "outcome": "decline", "reason": "51"}]}
    world = World(_sample(plan), trial_index=0)
    world.reset()
    world.step(SelectRoute(tool="select_route", acquirer_id="acq_a", method="visa"))
    r = world.step(SubmitAuthorization(tool="submit_authorization"))
    assert r.status is ToolStatus.DECLINED
    assert r.reason_code == "51"
    assert world.state is FsmState.DECLINED


def test_challenge_surfaces_requires_action_not_a_decline():
    plan = {"mode": "scripted", "draws": [{"attempt": 1, "outcome": "challenge"}]}
    world = World(_sample(plan), trial_index=0)
    world.reset()
    world.step(SelectRoute(tool="select_route", acquirer_id="acq_a", method="visa"))
    r = world.step(SubmitAuthorization(tool="submit_authorization"))
    assert r.status is ToolStatus.REQUIRES_ACTION
    assert r.challenge_token is not None
    assert r.reason_code is None  # 1A is never a decline
    assert world.state is FsmState.CHALLENGE


def test_handle_challenge_clears_to_approved():
    # The challenge resolves through its OWN cleared/failed space (G2); unpinned -> cleared.
    plan = {"mode": "scripted", "draws": [{"attempt": 1, "outcome": "challenge"}]}
    world = World(_sample(plan), trial_index=0)
    world.reset()
    world.step(SelectRoute(tool="select_route", acquirer_id="acq_a", method="visa"))
    r1 = world.step(SubmitAuthorization(tool="submit_authorization"))
    assert world.state is FsmState.CHALLENGE
    r2 = world.step(HandleChallenge(tool="handle_challenge", challenge_token=r1.challenge_token))
    assert r2.status is ToolStatus.APPROVED
    assert world.state is FsmState.APPROVED


def test_handle_challenge_failed_surfaces_a_decline():
    # A failed step-up resolves to a decline in the challenge outcome space — never a re-rolled challenge.
    plan = {
        "mode": "scripted",
        "draws": [{"attempt": 1, "outcome": "challenge"}],
        "challenge_draws": [{"attempt": 1, "outcome": "failed", "reason": "05"}],
    }
    world = World(_sample(plan), trial_index=0)
    world.reset()
    world.step(SelectRoute(tool="select_route", acquirer_id="acq_a", method="visa"))
    r1 = world.step(SubmitAuthorization(tool="submit_authorization"))
    assert world.state is FsmState.CHALLENGE
    r2 = world.step(HandleChallenge(tool="handle_challenge", challenge_token=r1.challenge_token))
    assert r2.status is ToolStatus.DECLINED
    assert r2.reason_code == "05"
    assert world.state is FsmState.DECLINED


def test_resubmit_raw_after_challenge_is_illegal_and_does_not_advance():
    plan = {"mode": "scripted", "draws": [{"attempt": 1, "outcome": "challenge"}]}
    world = World(_sample(plan), trial_index=0)
    world.reset()
    world.step(SelectRoute(tool="select_route", acquirer_id="acq_a", method="visa"))
    world.step(SubmitAuthorization(tool="submit_authorization"))
    assert world.state is FsmState.CHALLENGE
    r = world.step(SubmitAuthorization(tool="submit_authorization"))
    assert r.status is ToolStatus.ILLEGAL_ACTION
    assert world.state is FsmState.CHALLENGE  # state did NOT advance


def test_abort_from_declined_is_aborted_terminal():
    plan = {"mode": "scripted", "draws": [{"attempt": 1, "outcome": "decline", "reason": "43"}]}
    world = World(_sample(plan), trial_index=0)
    world.reset()
    world.step(SelectRoute(tool="select_route", acquirer_id="acq_a", method="visa"))
    world.step(SubmitAuthorization(tool="submit_authorization"))
    assert world.state is FsmState.DECLINED
    r = world.step(Abort(tool="abort", reason="hard decline"))
    assert r.status is ToolStatus.ABORTED
    assert world.state is FsmState.ABORTED
    assert world.is_agent_terminal()


def test_drain_deferred_resolves_settled_to_closed_identity():
    # Frozen M1 contract: drain_deferred() -> FsmState, identity SETTLED -> CLOSED in M2.
    plan = {"mode": "scripted", "draws": []}
    world = World(_sample(plan), trial_index=0)
    world.reset()
    world.step(SelectRoute(tool="select_route", acquirer_id="acq_a", method="visa"))
    world.step(SubmitAuthorization(tool="submit_authorization"))
    world.step(Capture(tool="capture"))
    assert world.state is FsmState.SETTLED
    assert world.drain_deferred() is FsmState.CLOSED


def test_drain_deferred_passes_aborted_through_unchanged():
    plan = {"mode": "scripted", "draws": [{"attempt": 1, "outcome": "decline", "reason": "43"}]}
    world = World(_sample(plan), trial_index=0)
    world.reset()
    world.step(SelectRoute(tool="select_route", acquirer_id="acq_a", method="visa"))
    world.step(SubmitAuthorization(tool="submit_authorization"))
    world.step(Abort(tool="abort", reason="hard decline"))
    assert world.drain_deferred() is FsmState.ABORTED
