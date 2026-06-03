"""The stable agent-facing contract: Observation (redacted), Action, ToolResponse."""

from __future__ import annotations

from decimal import Decimal
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from spar.simulator.enums import ToolStatus
from spar.simulator.mandates import CartMandate, IntentMandate, ScopeViolation


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---- Observation (redacted) ----

class ObsMethod(_Model):
    acquirer_id: str
    methods: list[str]
    geos: list[str]
    advertised_fee_bps: int
    observed_approval_band: Literal["low", "med", "high"]  # true approval prob stays hidden


class ObsCart(_Model):
    line_items: list[dict[str, Any]]
    subtotal: Decimal
    computed_tax: Decimal | None = None
    total: Decimal | None = None
    alternatives: list[dict[str, Any]] | None = None  # present only for semantic/underspecified samples
    merchant_note: str | None = None  # UNTRUSTED free text from the merchant; never an instruction


class ObsContext(_Model):
    buyer_geo: str
    elapsed_steps: int
    last_event: str | None = None
    last_reason_code: str | None = None
    attestation_present: bool = True
    attempt_counts: dict[str, int] = {}   # per-reason-code retry history
    retry_count: int = 0
    recent_events: list[str] = []
    authorization_only: bool = False  # True when only an authorization is warranted (do NOT capture; Void instead)


class Observation(_Model):
    mandate: IntentMandate
    cart: ObsCart
    methods: list[ObsMethod]
    context: ObsContext
    cart_mandate: CartMandate | None = None  # agent-visible assembled cart (payee + currency)


# ---- Action (tagged union over the 11 tools) ----

class SelectRoute(_Model):
    tool: Literal["select_route"]
    acquirer_id: str
    method: str


class ComputeTax(_Model):
    tool: Literal["compute_tax"]


class SubmitAuthorization(_Model):
    tool: Literal["submit_authorization"]
    idempotency_key: str | None = None


class HandleChallenge(_Model):
    tool: Literal["handle_challenge"]
    challenge_token: str


class Retry(_Model):
    tool: Literal["retry"]
    strategy: Literal["same", "different_method", "different_acquirer", "wait"]
    idempotency_key: str | None = None


class ModifyCart(_Model):
    tool: Literal["modify_cart"]
    field: str
    value: str


class RequestUserConfirmation(_Model):
    tool: Literal["request_user_confirmation"]
    reason: str


class Capture(_Model):
    tool: Literal["capture"]
    idempotency_key: str | None = None


class Void(_Model):
    tool: Literal["void"]
    idempotency_key: str | None = None


class Refund(_Model):
    tool: Literal["refund"]
    idempotency_key: str | None = None


class Abort(_Model):
    tool: Literal["abort"]
    reason: str


Action = Annotated[
    Union[
        SelectRoute, ComputeTax, SubmitAuthorization, HandleChallenge,
        Retry, ModifyCart, RequestUserConfirmation, Capture, Void, Refund, Abort,
    ],
    Field(discriminator="tool"),
]


# ---- ToolResponse (tagged union over ToolStatus) ----

class ToolResponse(_Model):
    status: ToolStatus
    reason_code: str | None = None       # set iff status == declined (never "1A")
    challenge_token: str | None = None   # set iff status == requires_action
    violation: ScopeViolation | None = None
    retry_fee_modeled: Decimal | None = None
    detail: dict[str, Any] = {}


_ACTION_ADAPTER: TypeAdapter[Action] = TypeAdapter(Action)


def parse_action(data: dict[str, Any]) -> Action:
    """Validate a raw dict into the correct Action member (discriminated by `tool`)."""
    return _ACTION_ADAPTER.validate_python(data)
