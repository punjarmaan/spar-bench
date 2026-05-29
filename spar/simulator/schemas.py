"""Dataset + world-config schemas (module 10 §8, module 30 §2). Validated at every boundary."""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from spar.simulator.enums import Axis, Difficulty, FsmState, IntentSpec
from spar.simulator.mandates import IntentMandate, MarketContext


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Acquirer(_Model):
    """A heterogeneous PSP. Exposed + hidden fields both live here; Observation redacts."""

    acquirer_id: str
    methods: list[str]
    supported_geos: list[str]
    advertised_fee_bps: int                       # EXPOSED
    observed_approval_band: Literal["low", "med", "high"]  # EXPOSED (F2) — noisy quantization
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
    max_steps: int = 30
    market_context: MarketContext | None = None   # semantic/underspecified samples only


class Gold(_Model):
    """Ground truth the grader reads (module 30 §2). All fields below are frozen here so
    downstream milestones consume them without re-editing this model."""

    correct_outcome: FsmState
    oracle_route: str | None = None
    acceptable_spend_band: tuple[Decimal, Decimal] | None = None
    overspend_ceiling: Decimal | None = None
    total: Decimal | None = None              # expected landed total (compliance_tax / stale_state, M5)
    trajectory: list[dict[str, Any]] | None = None      # gold action sequence replayed by GoldReplayAgent (M2)
    injection_demand: dict[str, Any] | None = None       # F11/F17 specific demanded deviation (M4 consumes it)
    must: list[str] = []
    must_not: list[str] = []


class Sample(_Model):
    """One dataset unit (one JSONL line)."""

    sample_id: str
    axis: Axis
    difficulty: Difficulty
    is_trap: bool
    intent_spec: IntentSpec = IntentSpec.EXPLICIT
    diamond: bool = False
    model_graded: bool = False
    seed: int
    canary: str
    world_config: WorldConfig
    mandate: IntentMandate
    policy_id: str
    gold: Gold
