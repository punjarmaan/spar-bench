"""Shared World fixture: an approvable single-intent world matching the lifecycle tests."""

from __future__ import annotations

from typing import Any

from spar.simulator.enums import Axis, Difficulty, FsmState, IntentSpec
from spar.simulator.schemas import Acquirer, Gold, Sample, WorldConfig
from spar.simulator.world import World
from tests.conftest import make_intent


def build_world(approval_prob: float = 1.0, **wc_over: Any) -> World:
    acq = Acquirer(
        acquirer_id="a1", methods=["card_visa"], supported_geos=["US"],
        advertised_fee_bps=200, observed_approval_band="high",
        true_fee_bps=200, approval_prob=approval_prob, reliability=1.0,
    )
    wc_kwargs: dict[str, Any] = dict(acquirers=[acq], settlement="sync", max_steps=20)
    wc_kwargs.update(wc_over)
    sample = Sample(
        sample_id="spar_post_purchase_lifecycle_0001", axis=Axis.POST_PURCHASE,
        difficulty=Difficulty.EASY, is_trap=False, intent_spec=IntentSpec.EXPLICIT,
        diamond=False, model_graded=False, seed=1, canary="spar:t",
        world_config=WorldConfig(**wc_kwargs),
        mandate=make_intent(allowed_instruments=["card_visa"], merchant_constraint=["acme"]),
        policy_id="default_v1", gold=Gold(correct_outcome=FsmState.CLOSED),
    )
    return World(sample, trial_index=0)
