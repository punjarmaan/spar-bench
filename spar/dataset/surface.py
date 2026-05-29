"""Per-seed surface randomization (module 30 §3): merchant names, amounts in a
band, geos, acquirer ids. Structure is fixed by difficulty elsewhere; this only
varies the cosmetic surface to kill memorization. All draws go through
`spar.simulator.rng.substream`, so the same (sample_id, seed) is byte-identical.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from spar.simulator.rng import SubStream, substream

_MERCHANTS: tuple[str, ...] = (
    "acme-store", "globex-shop", "initech-goods", "umbrella-market",
    "wayne-supplies", "stark-retail", "hooli-mart", "soylent-foods",
)
_GEOS: tuple[str, ...] = ("US", "CA", "GB", "DE", "FR")


@dataclass(frozen=True)
class Surface:
    merchant: str
    amount: Decimal
    buyer_geo: str
    acquirer_ids: tuple[str, ...]


def draw_surface(sample_id: str, *, seed: int, n_acquirers: int) -> Surface:
    """Deterministically draw cosmetic surface details for one sample."""
    # FRAUD sub-stream is repurposed here only as a fixed "surface" key; the choice
    # is arbitrary but frozen so draws never move. step=0, trial_index=0 for build.
    rng = substream(sample_id, seed=seed, trial_index=0, stream=SubStream.FRAUD, step=0)
    merchant = _MERCHANTS[int(rng.integers(0, len(_MERCHANTS)))]
    buyer_geo = _GEOS[int(rng.integers(0, len(_GEOS)))]
    cents = int(rng.integers(1000, 40001))  # 10.00 .. 400.00 inclusive
    amount = (Decimal(cents) / Decimal(100)).quantize(Decimal("0.01"))
    acquirer_ids = tuple(f"acq_{chr(ord('a') + i)}" for i in range(n_acquirers))
    return Surface(merchant=merchant, amount=amount, buyer_geo=buyer_geo,
                   acquirer_ids=acquirer_ids)
