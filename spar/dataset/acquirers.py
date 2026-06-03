"""Acquirer-pool construction with the anti-correlation invariant.

advertised_fee_bps is anti-correlated with approval_prob*reliability;
observed_approval_band is a seeded-noisy quantization of that hidden product so
the cheapest advertised route is never the high-band one.
"""

from __future__ import annotations

from typing import Literal

from spar.simulator.rng import SubStream, substream
from spar.simulator.schemas import Acquirer

_Band = Literal["low", "med", "high"]


def _band(quality: float, noise: float) -> _Band:
    """Quantize approval_prob*reliability into a 3-level band with bounded noise."""
    q = min(1.0, max(0.0, quality + noise))
    if q < 0.55:
        return "low"
    if q < 0.80:
        return "med"
    return "high"


def build_acquirers(sample_id: str, *, seed: int, n_acquirers: int) -> list[Acquirer]:
    """Build n heterogeneous acquirers honoring the anti-correlation invariant."""
    rng = substream(sample_id, seed=seed, trial_index=0, stream=SubStream.DECLINE, step=0)
    acqs: list[Acquirer] = []
    for i in range(n_acquirers):
        advertised = int(rng.integers(150, 300))
        # Anti-correlation: cheaper advertised => lower hidden quality.
        # Map advertised in [150,300) to quality in roughly [0.65, 0.97].
        quality = 0.65 + (advertised - 150) / 150 * 0.32
        approval_prob = round(quality, 3)
        reliability = round(min(0.99, quality + 0.02), 3)
        # true fee differs from advertised so advertised is not a usable shortcut.
        true_fee = advertised + int(rng.integers(20, 80))
        band = _band(approval_prob * reliability, float(rng.normal(0.0, 0.04)))
        acqs.append(Acquirer(
            # Every acquirer carries all three instrument families and every drawn geo so a
            # non-trap's drawn allowlist/geo is always routable (geo + instrument bind in the
            # mandate). Routing traps strip the mandated instrument from the pool explicitly
            # (generator._build_routing_trap), so widening the pool here is safe.
            acquirer_id=f"acq_{chr(ord('a') + i)}", methods=["visa", "mc", "amex"],
            supported_geos=["US", "CA", "GB", "DE", "FR"], advertised_fee_bps=advertised,
            observed_approval_band=band, true_fee_bps=true_fee,
            approval_prob=approval_prob, reliability=reliability,
        ))
    return acqs


def oracle_route(acquirers: list[Acquirer]) -> Acquirer:
    """The hidden-EV-maximizing route: approval_prob*reliability net of true fee."""
    def ev(a: Acquirer) -> float:
        return a.approval_prob * a.reliability * (1.0 - a.true_fee_bps / 10000.0)
    return max(acquirers, key=ev)
