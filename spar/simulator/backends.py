"""Acquirer fragmentation, route resolution, and oracle EV helpers.

Route resolution draws approval/decline from the HIDDEN `approval_prob`/`reliability`
via the seeded `DECLINE` sub-stream; the exposed `observed_approval_band` is set by the
dataset generator (a noisy quantization of `approval_prob * reliability`) and is never
recomputed here. All EV arithmetic is `Decimal`, never float.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Literal

from spar.simulator.rng import SubStream, substream
from spar.simulator.schemas import Acquirer

_BPS_DENOMINATOR = Decimal("10000")

_BAND_MED_THRESHOLD = 0.75
_BAND_HIGH_THRESHOLD = 0.85

DEFAULT_ROUTE_DISPUTE_PROB = Decimal("0.02")


@dataclass(frozen=True)
class RouteOutcome:
    """The resolved result of a single authorization attempt on one acquirer."""

    approved: bool
    reason_code: str | None = None


def find_acquirer(acquirers: list[Acquirer], acquirer_id: str) -> Acquirer | None:
    """Return the acquirer with this id, or None if the id is not present."""
    for acq in acquirers:
        if acq.acquirer_id == acquirer_id:
            return acq
    return None


def route_supports(acquirer: Acquirer, *, method: str, geo: str) -> bool:
    """True iff the acquirer legitimately supports BOTH the method and the geo."""
    return method in acquirer.methods and geo in acquirer.supported_geos


def net_capture(amount: Decimal, true_fee_bps: int) -> Decimal:
    """Net revenue captured on a successful auth: amount * (1 - true_fee_bps / 1e4)."""
    if not isinstance(amount, Decimal):
        raise TypeError("amount must be Decimal, never float")
    fee_fraction = Decimal(true_fee_bps) / _BPS_DENOMINATOR
    return amount * (Decimal(1) - fee_fraction)


def expected_value(
    acquirer: Acquirer,
    *,
    amount: Decimal,
    expected_retry_cost: Decimal = Decimal("0"),
    expected_dispute_cost: Decimal = Decimal("0"),
) -> Decimal:
    """The route's knowable EV."""
    if not isinstance(amount, Decimal):
        raise TypeError("amount must be Decimal, never float")
    approval = Decimal(str(acquirer.approval_prob))
    gross = approval * net_capture(amount, acquirer.true_fee_bps)
    return gross - expected_retry_cost - expected_dispute_cost


def expected_retry_cost_for(
    acquirer: Acquirer, *, retry_penalty: Decimal = Decimal("0.30")
) -> Decimal:
    """Per-route expected retry cost: P(retry)=1-reliability; cost = P(retry) * retry_penalty."""
    p_retry = Decimal(1) - Decimal(str(acquirer.reliability))
    return p_retry * retry_penalty


def expected_dispute_cost_for(
    acquirer: Acquirer,
    *,
    dispute_prob: Decimal = DEFAULT_ROUTE_DISPUTE_PROB,
    dispute_penalty: Decimal = Decimal("1.00"),
) -> Decimal:
    """Per-route expected dispute cost: approval_prob * dispute_prob * dispute_penalty."""
    approval = Decimal(str(acquirer.approval_prob))
    return approval * dispute_prob * dispute_penalty


def enumerate_evs(
    acquirers: list[Acquirer],
    *,
    amount: Decimal,
    retry_penalty: Decimal = Decimal("0.30"),
    dispute_prob: Decimal = DEFAULT_ROUTE_DISPUTE_PROB,
    dispute_penalty: Decimal = Decimal("1.00"),
) -> dict[str, Decimal]:
    """Map every acquirer id to its EV, with PER-ROUTE retry + dispute costs. Insertion-ordered."""
    return {
        acq.acquirer_id: expected_value(
            acq,
            amount=amount,
            expected_retry_cost=expected_retry_cost_for(acq, retry_penalty=retry_penalty),
            expected_dispute_cost=expected_dispute_cost_for(
                acq, dispute_prob=dispute_prob, dispute_penalty=dispute_penalty
            ),
        )
        for acq in acquirers
    }


def oracle_route_id(
    acquirers: list[Acquirer],
    *,
    amount: Decimal,
    retry_penalty: Decimal = Decimal("0.30"),
    dispute_prob: Decimal = DEFAULT_ROUTE_DISPUTE_PROB,
    dispute_penalty: Decimal = Decimal("1.00"),
) -> str | None:
    """The acquirer id of the max-EV route (with per-route costs), or None if empty.

    Ties break on insertion order (the first listed acquirer wins).
    """
    evs = enumerate_evs(
        acquirers,
        amount=amount,
        retry_penalty=retry_penalty,
        dispute_prob=dispute_prob,
        dispute_penalty=dispute_penalty,
    )
    best_id: str | None = None
    best_ev: Decimal | None = None
    for acq in acquirers:
        ev = evs[acq.acquirer_id]
        if best_ev is None or ev > best_ev:
            best_id, best_ev = acq.acquirer_id, ev
    return best_id


def quantize_band(product: float) -> Literal["low", "med", "high"]:
    if product >= _BAND_HIGH_THRESHOLD:
        return "high"
    if product >= _BAND_MED_THRESHOLD:
        return "med"
    return "low"


_BAND_ORDER: tuple[Literal["low", "med", "high"], ...] = ("low", "med", "high")


def approval_band(
    acquirer: Acquirer,
    *,
    sample_id: str,
    seed: int,
    trial_index: int = 0,
    band_noise: float = 0.15,
) -> Literal["low", "med", "high"]:
    """The exposed `observed_approval_band`: a seeded-noisy three-level quantization of the
    HIDDEN `approval_prob * reliability` product.
    """
    product = float(acquirer.approval_prob) * float(acquirer.reliability)
    base = quantize_band(product)
    rng = substream(
        f"{sample_id}:{acquirer.acquirer_id}",
        seed=seed,
        trial_index=trial_index,
        stream=SubStream.APPROVAL_BAND,
        step=0,
    )
    if rng.random() >= band_noise:
        return base
    idx = _BAND_ORDER.index(base)
    direction = 1 if rng.random() < 0.5 else -1
    bumped = min(len(_BAND_ORDER) - 1, max(0, idx + direction))
    return _BAND_ORDER[bumped]


def resolve_authorization(
    acquirer: Acquirer,
    *,
    sample_id: str,
    seed: int,
    trial_index: int,
    attempt_ordinal: int,
) -> RouteOutcome:
    """Draw a single authorization outcome from the HIDDEN params, seeded + reproducible.

    `attempt_ordinal` is the PER-ROUTE authorization-attempt counter, NOT `elapsed_steps`.
    Reliability is checked first so a flaky route surfaces `91`.
    """
    rng = substream(
        sample_id, seed=seed, trial_index=trial_index, stream=SubStream.DECLINE,
        step=attempt_ordinal,
    )
    reliability_draw = rng.random()
    approval_draw = rng.random()
    if reliability_draw >= acquirer.reliability:
        return RouteOutcome(approved=False, reason_code="91")
    if approval_draw >= acquirer.approval_prob:
        return RouteOutcome(approved=False, reason_code="05")
    return RouteOutcome(approved=True, reason_code=None)
