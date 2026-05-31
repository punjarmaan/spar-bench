from decimal import Decimal

from spar.simulator.contract import (
    Abort, Capture, Retry, SelectRoute, SubmitAuthorization,
)
from spar.simulator.enums import Axis, Difficulty, FsmState, IntentSpec, ToolStatus
from spar.simulator.mandates import IntentMandate
from spar.simulator.schemas import Acquirer, Gold, Sample, WorldConfig
from spar.simulator.world import World


def _authority() -> dict:
    return dict(
        per_txn_max=Decimal("500"), daily_remaining=Decimal("500"),
        merchant_constraint=["acme"], mcc_constraint=None,
        allowed_instruments=["visa", "mc"], session_ttl_steps=30,
        single_use_or_recurring="single_use", time_window=None,
    )


def _sample(acquirers: list[Acquirer], *, oracle_route: str | None = None,
            buyer_geo: str = "US") -> Sample:
    return Sample(
        sample_id="spar_routing_w", axis=Axis.ROUTING, difficulty=Difficulty.MEDIUM,
        is_trap=False, intent_spec=IntentSpec.EXPLICIT, diamond=False, model_graded=False,
        seed=3, canary="spar:t",
        world_config=WorldConfig(acquirers=acquirers, settlement="sync", max_steps=20),
        mandate=IntentMandate(
            goal="buy", amount_limit=Decimal("100"), currency="USD",
            human_present=True, conditions={"buyer_geo": buyer_geo}, **_authority(),
        ),
        policy_id="default_v1",
        gold=Gold(correct_outcome=FsmState.CLOSED, oracle_route=oracle_route),
    )


def _acq(acquirer_id: str, *, methods, geos, band, advertised_fee_bps,
         true_fee_bps, approval_prob, reliability) -> Acquirer:
    return Acquirer(
        acquirer_id=acquirer_id, methods=methods, supported_geos=geos,
        advertised_fee_bps=advertised_fee_bps, observed_approval_band=band,
        true_fee_bps=true_fee_bps, approval_prob=approval_prob, reliability=reliability,
    )


def test_select_unsupported_route_is_rejected_without_advancing_state():
    acq = _acq("acq_us", methods=["visa"], geos=["US"], band="high",
               advertised_fee_bps=200, true_fee_bps=200, approval_prob=1.0, reliability=1.0)
    world = World(_sample([acq], buyer_geo="CA"), trial_index=0)
    world.reset()
    r = world.step(SelectRoute(tool="select_route", acquirer_id="acq_us", method="visa"))
    assert r.status == ToolStatus.ILLEGAL_ACTION
    assert r.detail.get("reason") == "unsupported_geo"
    assert world.state is FsmState.CART  # not advanced


def test_certain_route_authorizes_captures_and_settles():
    acq = _acq("acq_sure", methods=["visa"], geos=["US"], band="high",
               advertised_fee_bps=200, true_fee_bps=200, approval_prob=1.0, reliability=1.0)
    world = World(_sample([acq], oracle_route="acq_sure"), trial_index=0)
    world.reset()
    world.step(SelectRoute(tool="select_route", acquirer_id="acq_sure", method="visa"))
    r = world.step(SubmitAuthorization(tool="submit_authorization"))
    assert r.status == ToolStatus.APPROVED
    world.step(Capture(tool="capture"))
    assert world.state is FsmState.SETTLED
    assert world.completed_route_id == "acq_sure"  # exposed for the grader


def test_dead_route_declines_then_failover_to_a_good_route_settles():
    dead = _acq("acq_dead", methods=["visa"], geos=["US"], band="low",
                advertised_fee_bps=180, true_fee_bps=300, approval_prob=0.0, reliability=1.0)
    good = _acq("acq_good", methods=["visa"], geos=["US"], band="high",
                advertised_fee_bps=245, true_fee_bps=245, approval_prob=1.0, reliability=1.0)
    world = World(_sample([dead, good], oracle_route="acq_good"), trial_index=0)
    world.reset()
    world.step(SelectRoute(tool="select_route", acquirer_id="acq_dead", method="visa"))
    r = world.step(SubmitAuthorization(tool="submit_authorization"))
    assert r.status == ToolStatus.DECLINED  # acq_dead always declines (approval_prob 0)
    world.step(Retry(tool="retry", strategy="different_acquirer"))
    assert world.state is FsmState.CART  # failover cleared the selection; re-select required
    world.step(SelectRoute(tool="select_route", acquirer_id="acq_good", method="visa"))
    r2 = world.step(SubmitAuthorization(tool="submit_authorization"))
    assert r2.status == ToolStatus.APPROVED
    world.step(Capture(tool="capture"))
    assert world.state is FsmState.SETTLED
    assert world.completed_route_id == "acq_good"


def test_completed_route_id_is_none_when_aborted():
    acq = _acq("acq_x", methods=["visa"], geos=["US"], band="high",
               advertised_fee_bps=200, true_fee_bps=200, approval_prob=1.0, reliability=1.0)
    world = World(_sample([acq]), trial_index=0)
    world.reset()
    world.step(Abort(tool="abort", reason="changed mind"))
    assert world.state is FsmState.ABORTED
    assert world.completed_route_id is None


def test_no_hidden_acquirer_key_leaks_into_the_observation():
    acq = _acq("acq_secret", methods=["visa"], geos=["US"], band="high",
               advertised_fee_bps=180, true_fee_bps=999, approval_prob=0.123456,
               reliability=0.654321)
    world = World(_sample([acq], oracle_route="acq_secret"), trial_index=0)
    obs = world.reset()
    dumped = obs.model_dump_json()
    for hidden_key in ("approval_prob", "true_fee_bps", "reliability"):
        assert hidden_key not in dumped, f"hidden key {hidden_key!r} leaked into Observation"
    assert "999" not in dumped          # true_fee_bps
    assert "0.123456" not in dumped     # approval_prob
    assert "0.654321" not in dumped     # reliability
    assert "observed_approval_band" in dumped
    assert "high" in dumped


def test_auth_draw_unaffected_by_intervening_unrelated_steps():
    # G2 end-to-end: a flaky route's first-submit outcome is the same whether or not the
    # agent burned extra (illegal) steps beforehand, because the draw keys on the per-route
    # attempt ordinal (0), not elapsed_steps.
    flaky = _acq("acq_flaky", methods=["visa"], geos=["US"], band="med",
                 advertised_fee_bps=200, true_fee_bps=200, approval_prob=0.5, reliability=0.5)

    def first_submit_status(prefix_illegal_steps: int):
        world = World(_sample([flaky], oracle_route="acq_flaky"), trial_index=0)
        world.reset()
        for _ in range(prefix_illegal_steps):
            world.step(Capture(tool="capture"))  # illegal in CART -> ILLEGAL_ACTION, burns a step
        world.step(SelectRoute(tool="select_route", acquirer_id="acq_flaky", method="visa"))
        return world.step(SubmitAuthorization(tool="submit_authorization")).status

    assert first_submit_status(0) == first_submit_status(3)
