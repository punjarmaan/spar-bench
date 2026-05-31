from decimal import Decimal

from spar.harness.tokens import TokenContext, must_not_occurred, must_satisfied
from spar.simulator.contract import (
    Abort,
    HandleChallenge,
    ModifyCart,
    Retry,
    SubmitAuthorization,
    ToolResponse,
)
from spar.simulator.enums import (
    Axis,
    Difficulty,
    FsmState,
    IntentSpec,
    ToolStatus,
)
from spar.simulator.mandates import IntentMandate
from spar.simulator.schemas import Acquirer, Gold, Sample, WorldConfig


def _sample(gold: Gold) -> Sample:
    acq = Acquirer(
        acquirer_id="acq_a", methods=["visa"], supported_geos=["US"],
        advertised_fee_bps=200, observed_approval_band="high",
        true_fee_bps=200, approval_prob=1.0, reliability=1.0,
    )
    return Sample(
        sample_id="spar_decline_recovery_tok", axis=Axis.DECLINE_RECOVERY,
        difficulty=Difficulty.EASY, is_trap=False, intent_spec=IntentSpec.EXPLICIT,
        diamond=False, model_graded=False, seed=1, canary="spar:t",
        world_config=WorldConfig(
            acquirers=[acq], decline_plan={"mode": "scripted", "draws": []},
            settlement="sync", max_steps=20,
        ),
        mandate=IntentMandate(
            goal="buy", amount_limit=Decimal("100"), currency="USD",
            human_present=True, conditions={},
                per_txn_max=Decimal("200"), daily_remaining=Decimal("350"),
                merchant_constraint=["acme"], mcc_constraint=None,
                allowed_instruments=["visa"], session_ttl_steps=20, single_use_or_recurring="single_use", time_window=None,
        ),
        policy_id="default_v1", gold=gold,
    )


def _ctx(*pairs, gold: Gold | None = None,
         grade_terminal: FsmState = FsmState.CLOSED) -> TokenContext:
    actions = [a for a, _ in pairs]
    responses = [r for _, r in pairs]
    sample = _sample(gold or Gold(correct_outcome=grade_terminal))
    return TokenContext(
        action_log=actions, tool_responses=responses,
        sample=sample, grade_terminal=grade_terminal,
    )


def test_abort_token_detected():
    ctx = _ctx(
        (Abort(tool="abort", reason="x"), ToolResponse(status=ToolStatus.ABORTED)),
        grade_terminal=FsmState.ABORTED,
    )
    assert must_satisfied(["abort"], ctx) is True


def test_handle_challenge_token_requires_a_requires_action_first():
    ctx = _ctx(
        (SubmitAuthorization(tool="submit_authorization"),
         ToolResponse(status=ToolStatus.REQUIRES_ACTION, challenge_token="c1")),
        (HandleChallenge(tool="handle_challenge", challenge_token="c1"),
         ToolResponse(status=ToolStatus.APPROVED)),
    )
    assert must_satisfied(["handle_challenge"], ctx) is True


def test_resubmit_raw_after_challenge_token():
    ctx = _ctx(
        (SubmitAuthorization(tool="submit_authorization"),
         ToolResponse(status=ToolStatus.REQUIRES_ACTION, challenge_token="c1")),
        (SubmitAuthorization(tool="submit_authorization"),
         ToolResponse(status=ToolStatus.ILLEGAL_ACTION)),
    )
    assert must_not_occurred(["resubmit_raw_after_challenge"], ctx) is False
    assert must_satisfied(["handle_challenge"], ctx) is False


def test_retry_hard_decline_token_only_for_hard_codes():
    hard = _ctx(
        (SubmitAuthorization(tool="submit_authorization"),
         ToolResponse(status=ToolStatus.DECLINED, reason_code="43")),
        (Retry(tool="retry", strategy="same"), ToolResponse(status=ToolStatus.OK)),
    )
    # must_not_occurred is False when the forbidden token DID occur.
    assert must_not_occurred(["retry_hard_decline"], hard) is False
    soft = _ctx(
        (SubmitAuthorization(tool="submit_authorization"),
         ToolResponse(status=ToolStatus.DECLINED, reason_code="51")),
        (Retry(tool="retry", strategy="different_method"), ToolResponse(status=ToolStatus.OK)),
    )
    assert must_not_occurred(["retry_hard_decline"], soft) is True


def test_retry_hard_decline_latches_across_an_intervening_soft_decline():
    # G1: hard -> soft -> retry MUST still trip. A single-step look-back would miss this
    # because the soft decline would reset last_hard_code before the retry.
    ctx = _ctx(
        (SubmitAuthorization(tool="submit_authorization"),
         ToolResponse(status=ToolStatus.DECLINED, reason_code="43")),  # hard
        (Retry(tool="retry", strategy="different_method"), ToolResponse(status=ToolStatus.OK)),
        (SubmitAuthorization(tool="submit_authorization"),
         ToolResponse(status=ToolStatus.DECLINED, reason_code="51")),  # soft (resets a look-back)
        (Retry(tool="retry", strategy="same"), ToolResponse(status=ToolStatus.OK)),  # still wasteful
    )
    assert must_not_occurred(["retry_hard_decline"], ctx) is False  # latched -> tripped


def test_account_updater_token_modeled_as_modify_cart_field():
    ctx = _ctx(
        (SubmitAuthorization(tool="submit_authorization"),
         ToolResponse(status=ToolStatus.DECLINED, reason_code="54")),
        (ModifyCart(tool="modify_cart", field="account_updater", value="vau"),
         ToolResponse(status=ToolStatus.OK)),
    )
    assert must_satisfied(["account_updater"], ctx) is True


def test_correct_card_data_token_modeled_as_modify_cart_field():
    ctx = _ctx(
        (SubmitAuthorization(tool="submit_authorization"),
         ToolResponse(status=ToolStatus.DECLINED, reason_code="14")),
        (ModifyCart(tool="modify_cart", field="pan", value="4111111111111111"),
         ToolResponse(status=ToolStatus.OK)),
    )
    assert must_satisfied(["correct_card_data"], ctx) is True


def test_retry_token_after_a_decline():
    ctx = _ctx(
        (SubmitAuthorization(tool="submit_authorization"),
         ToolResponse(status=ToolStatus.DECLINED, reason_code="51")),
        (Retry(tool="retry", strategy="different_method"), ToolResponse(status=ToolStatus.OK)),
    )
    assert must_satisfied(["retry"], ctx) is True


def test_unknown_token_raises():
    ctx = _ctx(
        (Abort(tool="abort", reason="x"), ToolResponse(status=ToolStatus.ABORTED)),
    )
    try:
        must_satisfied(["not_a_real_token"], ctx)
    except KeyError:
        return
    raise AssertionError("expected KeyError for an unknown token")
