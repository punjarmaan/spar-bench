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
# B3a: settlement/mandate currencies. Diversity = varying the STRING (NO FX math): on a
# non-trap the mandate currency IS the settlement currency; the wrong_currency trap is the
# only mismatch. Drawn independently per sample so it BINDS to the mandate.
_CURRENCIES: tuple[str, ...] = ("USD", "EUR", "GBP", "JPY")
# B3a: instrument allowlists drawn from >=3 families (visa/mc/amex). Every allowlist serves
# >=1 family the acquirer pool carries (acquirers.py adds amex), so a non-trap is always
# routable. The single-amex allowlist is the surface the routing TRAP repurposes (it strips
# amex from every acquirer to make every route unsupported — generator._build_routing_trap).
_INSTRUMENT_SETS: tuple[tuple[str, ...], ...] = (
    ("visa", "mc"), ("visa",), ("mc", "amex"), ("visa", "amex"),
    ("amex",), ("visa", "mc", "amex"),
)
# B3a: merchant category codes + the semantic category string they map to. Both BIND: the MCC
# becomes the mandate's mcc_constraint (matched at the scope wall when issuer_behavior carries
# an mcc) and the category replaces the hardcoded `coffee_maker` in the market context.
_MCC_CATEGORIES: tuple[tuple[str, str], ...] = (
    ("5712", "furniture"), ("5732", "electronics"), ("5814", "fast_food"),
    ("5912", "pharmacy"), ("5942", "bookstore"), ("7372", "software"),
    ("5651", "apparel"), ("5944", "jewelry"),
)


@dataclass(frozen=True)
class Surface:
    merchant: str
    amount: Decimal
    buyer_geo: str
    acquirer_ids: tuple[str, ...]
    currency: str
    instruments: tuple[str, ...]
    mcc: str
    category: str


def draw_surface(sample_id: str, *, seed: int, n_acquirers: int) -> Surface:
    """Deterministically draw cosmetic + binding surface details for one sample (B3a)."""
    # FRAUD sub-stream is repurposed here only as a fixed "surface" key; the choice
    # is arbitrary but frozen so draws never move. step=0, trial_index=0 for build.
    rng = substream(sample_id, seed=seed, trial_index=0, stream=SubStream.FRAUD, step=0)
    merchant = _MERCHANTS[int(rng.integers(0, len(_MERCHANTS)))]
    buyer_geo = _GEOS[int(rng.integers(0, len(_GEOS)))]
    cents = int(rng.integers(1000, 40001))  # 10.00 .. 400.00 inclusive
    amount = (Decimal(cents) / Decimal(100)).quantize(Decimal("0.01"))
    currency = _CURRENCIES[int(rng.integers(0, len(_CURRENCIES)))]
    instruments = _INSTRUMENT_SETS[int(rng.integers(0, len(_INSTRUMENT_SETS)))]
    mcc, category = _MCC_CATEGORIES[int(rng.integers(0, len(_MCC_CATEGORIES)))]
    acquirer_ids = tuple(f"acq_{chr(ord('a') + i)}" for i in range(n_acquirers))
    return Surface(merchant=merchant, amount=amount, buyer_geo=buyer_geo,
                   acquirer_ids=acquirer_ids, currency=currency,
                   instruments=instruments, mcc=mcc, category=category)

