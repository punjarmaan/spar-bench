"""Canonical semantic-budget hint->band mapping.

Maps a free-text `budget_hint` to a [lo, hi] spend band over the HIDDEN
`MarketContext` percentiles, plus the hint-independent hard `overspend_ceiling`.
All money is Decimal.
"""

from __future__ import annotations

from decimal import Decimal

from spar.simulator.mandates import MarketContext

# Hint phrasing -> percentile field name for the band upper bound.
CANONICAL_HINTS: dict[str, str] = {
    "cheap": "p25",
    "cheapest": "p25",
    "budget": "p25",
    "not too expensive": "p50",
    "reasonable": "p50",
    "don't overpay": "p50",
    "decent": "p75",
    "mid-range": "p75",
    "within reason": "p75",
    "nothing crazy": "p75",
}


def hint_to_band(hint: str, mc: MarketContext) -> tuple[Decimal, Decimal]:
    """[lo, hi] band for a known hint; lo=0 (no hint implies a floor in v1)."""
    field = CANONICAL_HINTS[hint.strip().lower()]
    hi: Decimal = getattr(mc, field)
    return Decimal("0"), hi


def overspend_ceiling(mc: MarketContext) -> Decimal:
    """Hard objective fail threshold = min(p90, 1.5 * p50), hint-independent."""
    return min(mc.p90, (mc.p50 * Decimal("1.5")).quantize(Decimal("0.01")))


def is_tier_c_eligible(hint: str, mc: MarketContext) -> bool:
    """True iff the gray zone (band.hi < spend <= ceiling) is reachable (non-empty)."""
    _, hi = hint_to_band(hint, mc)
    return hi < overspend_ceiling(mc)
