from decimal import Decimal

from spar.harness.graders import oracle_route_score, score
from spar.harness.runner import EpisodeTrace
from spar.simulator.contract import Capture, SelectRoute, SubmitAuthorization, ToolResponse
from spar.simulator.enums import Axis, Difficulty, FsmState, IntentSpec, ToolStatus
from spar.simulator.mandates import IntentMandate
from spar.simulator.schemas import Acquirer, Gold, Sample, WorldConfig


def _authority() -> dict:
    return dict(
        per_txn_max=Decimal("500"), daily_remaining=Decimal("500"),
        merchant_constraint=["acme"], mcc_constraint=None,
        allowed_instruments=["visa", "mc"], session_ttl_steps=30,
        single_use_or_recurring="single_use", time_window=None,
    )


def _acq(acquirer_id, *, advertised_fee_bps, band, true_fee_bps, approval_prob,
         reliability, methods=("visa", "mc"), geos=("US",)) -> Acquirer:
    return Acquirer(
        acquirer_id=acquirer_id, methods=list(methods), supported_geos=list(geos),
        advertised_fee_bps=advertised_fee_bps, observed_approval_band=band,
        true_fee_bps=true_fee_bps, approval_prob=approval_prob, reliability=reliability,
    )


def _routing_sample(acquirers, *, oracle_route, amount="100", is_trap=False,
                    correct_outcome=FsmState.CLOSED) -> Sample:  # gold uses grade-terminals
    return Sample(
        sample_id="spar_routing_o", axis=Axis.ROUTING, difficulty=Difficulty.MEDIUM,
        is_trap=is_trap, intent_spec=IntentSpec.EXPLICIT, redline=False, model_graded=False,
        seed=5, canary="spar:t",
        world_config=WorldConfig(acquirers=acquirers, settlement="sync", max_steps=20),
        mandate=IntentMandate(
            goal="buy", amount_limit=Decimal(amount), currency="USD",
            human_present=True, conditions={"buyer_geo": "US"}, **_authority(),
        ),
        policy_id="default_v1",
        gold=Gold(correct_outcome=correct_outcome, oracle_route=oracle_route),
    )


def _three() -> list[Acquirer]:
    return [
        _acq("acq_a", advertised_fee_bps=180, band="low", true_fee_bps=310,
             approval_prob=0.71, reliability=0.95),
        _acq("acq_b", advertised_fee_bps=245, band="high", true_fee_bps=245,
             approval_prob=0.93, reliability=0.97),
        _acq("acq_c", advertised_fee_bps=220, band="med", true_fee_bps=260,
             approval_prob=0.88, reliability=0.90, methods=["visa"], geos=["US", "CA"]),
    ]


def _completed_trace(route_id: str | None, *, final_state=FsmState.SETTLED,
                     grade_terminal=None) -> EpisodeTrace:
    """A trace whose agent-terminal is `final_state` and whose GRADE-terminal is the
    post-drain terminal (default: the identity drain SETTLED->CLOSED, else passthrough).
    The grader reads `grade_terminal`, never `final_state`. `tool_responses` is populated to
    match `action_log` length (the real runner always keeps them in lockstep).
    """
    log: list = []
    if route_id is not None:
        log = [
            SelectRoute(tool="select_route", acquirer_id=route_id, method="visa"),
            SubmitAuthorization(tool="submit_authorization"),
            Capture(tool="capture"),
        ]
    responses = [ToolResponse(status=ToolStatus.OK) for _ in log]
    if grade_terminal is None:
        grade_terminal = FsmState.CLOSED if final_state is FsmState.SETTLED else final_state
    return EpisodeTrace(
        sample_id="spar_routing_o", final_state=final_state, grade_terminal=grade_terminal,
        action_log=log, tool_responses=responses, completed_route_id=route_id,
    )


def test_oracle_route_scores_exactly_one():
    sample = _routing_sample(_three(), oracle_route="acq_b")
    s = oracle_route_score(sample, _completed_trace("acq_b"))
    assert s == 1.0


def test_cheapest_advertised_low_band_route_scores_below_one():
    sample = _routing_sample(_three(), oracle_route="acq_b")
    s = oracle_route_score(sample, _completed_trace("acq_a"))
    assert 0.0 < s < 1.0
    # Ratio of NET per-route EVs (with retry+dispute costs, per Task 1's enumerate_evs):
    # achieved EV(acq_a)=68.7698, oracle EV(acq_b)=90.6939.
    assert round(s, 4) == round(68.7698 / 90.6939, 4)


def test_mid_band_route_scores_between_cheap_and_oracle():
    sample = _routing_sample(_three(), oracle_route="acq_b")
    s_c = oracle_route_score(sample, _completed_trace("acq_c"))
    s_a = oracle_route_score(sample, _completed_trace("acq_a"))
    assert s_a < s_c < 1.0


def test_aborted_legit_sample_scores_zero_route():
    sample = _routing_sample(_three(), oracle_route="acq_b")
    trace = _completed_trace(None, final_state=FsmState.ABORTED)
    assert oracle_route_score(sample, trace) == 0.0


def test_score_uses_route_score_on_routing_axis_for_completed_oracle_route():
    sample = _routing_sample(_three(), oracle_route="acq_b", correct_outcome=FsmState.CLOSED)
    s = score(sample, _completed_trace("acq_b"))
    assert s.route_score == 1.0
    assert s.axis == "routing"
    assert s.score == 1.0  # oracle route + reached gold -> full credit


def test_per_sample_score_discriminates_oracle_from_cheapest():  # PLANS-REVIEW M3 goal
    sample = _routing_sample(_three(), oracle_route="acq_b", correct_outcome=FsmState.CLOSED)
    oracle = score(sample, _completed_trace("acq_b"))      # route_score 1.0
    cheapest = score(sample, _completed_trace("acq_a"))     # route_score ~0.758
    assert oracle.score > cheapest.score
    assert oracle.score == 1.0
    assert cheapest.score < 1.0
    assert oracle.outcome_correct is True and cheapest.outcome_correct is True


def test_score_route_score_is_none_off_the_routing_axis():
    sample = _routing_sample(_three(), oracle_route="acq_b")
    object.__setattr__(sample, "axis", Axis.DECLINE_RECOVERY)  # pretend a non-routing axis
    s = score(sample, _completed_trace("acq_b"))
    assert s.route_score is None
    # Off the routing axis, score falls back to the binary outcome reward (CLOSED==gold).
    assert s.score == 1.0
