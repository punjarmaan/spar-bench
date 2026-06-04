from decimal import Decimal

from spar.simulator.enums import FsmState, ToolStatus
from spar.simulator.contract import SelectRoute, SubmitAuthorization, Capture, Abort
from spar.simulator.world import World
from spar.simulator.schemas import Sample, WorldConfig, Gold, Acquirer
from spar.simulator.enums import Axis, Difficulty, IntentSpec
from spar.simulator.mandates import IntentMandate


def _sample() -> Sample:
    acq = Acquirer(acquirer_id="acq_a", methods=["visa"], supported_geos=["US"],
                   advertised_fee_bps=200, observed_approval_band="high",
                   true_fee_bps=200, approval_prob=1.0, reliability=1.0)
    return Sample(
        sample_id="spar_routing_0001", axis=Axis.ROUTING, difficulty=Difficulty.EASY,
        is_trap=False, intent_spec=IntentSpec.EXPLICIT, redline=False, model_graded=False,
        seed=1, canary="spar:t", world_config=WorldConfig(acquirers=[acq], settlement="sync", max_steps=10),
        mandate=IntentMandate(goal="buy", amount_limit=Decimal("100"), currency="USD",
                              human_present=True, conditions={},
                              per_txn_max=Decimal("200"), daily_remaining=Decimal("350"),
                              merchant_constraint=["acme"], mcc_constraint=None,
                              allowed_instruments=["visa"], session_ttl_steps=10,
                              single_use_or_recurring="single_use", time_window=None),
        policy_id="default_v1", gold=Gold(correct_outcome=FsmState.CLOSED),
    )


def test_happy_path_reaches_settled_agent_terminal():
    world = World(_sample(), trial_index=0)
    world.reset()
    world.step(SelectRoute(tool="select_route", acquirer_id="acq_a", method="visa"))
    r = world.step(SubmitAuthorization(tool="submit_authorization"))
    assert r.status == ToolStatus.APPROVED
    world.step(Capture(tool="capture"))
    assert world.state is FsmState.SETTLED  # F3: agent-terminal
    assert world.is_agent_terminal()
    # The grade-terminal is resolved by the drain: SETTLED -> CLOSED (identity in M1).
    assert world.drain_deferred() is FsmState.CLOSED


def test_abort_is_terminal():
    world = World(_sample(), trial_index=0)
    world.reset()
    world.step(Abort(tool="abort", reason="x"))
    assert world.state is FsmState.ABORTED and world.is_agent_terminal()
