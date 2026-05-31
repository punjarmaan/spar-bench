"""M7 prereq: the select_oracle_route routing `must` token (completed == gold.oracle_route)."""

from __future__ import annotations

from decimal import Decimal

from spar.harness.tokens import TokenContext, must_satisfied
from spar.simulator.enums import Axis, Difficulty, FsmState, IntentSpec
from spar.simulator.mandates import IntentMandate
from spar.simulator.schemas import Acquirer, Gold, Sample, WorldConfig


def _sample(oracle: str | None) -> Sample:
    acq = Acquirer(acquirer_id="acq_a", methods=["visa"], supported_geos=["US"],
                   advertised_fee_bps=200, observed_approval_band="high",
                   true_fee_bps=200, approval_prob=1.0, reliability=1.0)
    return Sample(
        sample_id="spar_routing_tok", axis=Axis.ROUTING, difficulty=Difficulty.EASY,
        is_trap=False, intent_spec=IntentSpec.EXPLICIT, diamond=False, model_graded=False,
        seed=1, canary="spar:t",
        world_config=WorldConfig(acquirers=[acq], settlement="sync"),
        mandate=IntentMandate(goal="buy", amount_limit=Decimal("100"), currency="USD",
                              human_present=True, conditions={}, 
                                  per_txn_max=Decimal("200"), daily_remaining=Decimal("350"),
                                  merchant_constraint=["acme"], mcc_constraint=None,
                                  allowed_instruments=["visa"], session_ttl_steps=20, single_use_or_recurring="single_use", time_window=None),
        policy_id="default_v1", gold=Gold(correct_outcome=FsmState.CLOSED, oracle_route=oracle),
    )


def _ctx(sample: Sample, completed: str | None) -> TokenContext:
    return TokenContext(action_log=[], tool_responses=[], sample=sample,
                        grade_terminal=FsmState.CLOSED,
                        extra={"completed_route_id": completed})


def test_select_oracle_route_satisfied_on_oracle():
    s = _sample(oracle="acq_b")
    assert must_satisfied(["select_oracle_route"], _ctx(s, "acq_b")) is True


def test_select_oracle_route_unsatisfied_off_oracle_or_no_completion():
    s = _sample(oracle="acq_b")
    assert must_satisfied(["select_oracle_route"], _ctx(s, "acq_a")) is False
    assert must_satisfied(["select_oracle_route"], _ctx(s, None)) is False
