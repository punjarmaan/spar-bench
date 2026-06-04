"""Dataset + world-config schemas. Validated at every boundary."""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from spar.simulator.enums import Axis, Difficulty, FsmState, IntentSpec, ViolationClass
from spar.simulator.mandates import IntentMandate, MarketContext


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Acquirer(_Model):
    """A heterogeneous PSP. Exposed + hidden fields both live here; Observation redacts."""

    acquirer_id: str
    methods: list[str]
    supported_geos: list[str]
    advertised_fee_bps: int                       # EXPOSED
    observed_approval_band: Literal["low", "med", "high"]  # EXPOSED — noisy quantization
    true_fee_bps: int                             # HIDDEN
    approval_prob: float                          # HIDDEN
    reliability: float                            # HIDDEN


class WorldConfig(_Model):
    """The full (mostly hidden) parameterization of one episode's world."""

    acquirers: list[Acquirer] = []
    issuer_behavior: dict[str, Any] = {}
    decline_plan: dict[str, Any] = {}
    fraud_engine: dict[str, Any] = {}
    settlement: Literal["sync", "async"] = "sync"
    dispute: dict[str, Any] = {}
    responder: bool = False   # when True, request_user_confirmation is non-terminal
    max_steps: int = 30
    market_context: MarketContext | None = None   # semantic/underspecified samples only
    # The concrete cart total the agent is shown + charged (the intended `amount`, kept under
    # per_txn_max for in-scope samples). When None the world falls back to mandate.amount_limit
    # (legacy/hand-authored gold). Procedural samples set this so the shown cart is honest.
    cart_total: Decimal | None = None


class Gold(_Model):
    """Ground truth the grader reads."""

    correct_outcome: FsmState
    oracle_route: str | None = None
    acceptable_spend_band: tuple[Decimal, Decimal] | None = None
    overspend_ceiling: Decimal | None = None
    total: Decimal | None = None              # expected landed total (compliance_tax / stale_state)
    trajectory: list[dict[str, Any]] | None = None      # gold action sequence replayed by the gold agent
    injection_demand: dict[str, Any] | None = None       # specific demanded deviation
    must: list[str] = []
    must_not: list[str] = []
    expected_violations: list[ViolationClass] = []   # applicability (derived at build)
    violated_field: str | None = None                # the ONLY hand-authored consent field
    intended_violation_class: ViolationClass | None = None  # the ONE catastrophic class this trap is
                                                            # built to elicit (per-class rate
                                                            # attribution). None for non-catastrophic.


class Sample(_Model):
    """One dataset unit (one JSONL line)."""

    sample_id: str
    axis: Axis
    difficulty: Difficulty
    is_trap: bool
    intent_spec: IntentSpec = IntentSpec.EXPLICIT
    redline: bool = False
    split: str | None = None   # build-stamped membership ("lite"|"main"|"redline"|"probe"); None
                               # for ad-hoc/toy/gold samples. Dropped by public projection.
                               # "probe" = derived stratified ~50% subsample of main (not canonical).
    model_graded: bool = False
    seed: int
    canary: str
    world_config: WorldConfig
    mandate: IntentMandate
    policy_id: str
    gold: Gold
