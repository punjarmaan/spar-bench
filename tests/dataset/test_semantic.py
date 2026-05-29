from decimal import Decimal

import pytest

from spar.simulator.mandates import MarketContext
from spar.dataset.semantic import (
    CANONICAL_HINTS, hint_to_band, overspend_ceiling, is_tier_c_eligible,
)


def _mc() -> MarketContext:
    return MarketContext(
        category="coffee_maker", currency="USD",
        p25=Decimal("35.00"), p50=Decimal("60.00"),
        p75=Decimal("95.00"), p90=Decimal("140.00"),
    )


def test_canonical_hint_phrasings_cover_three_bands():
    assert CANONICAL_HINTS["cheap"] == "p25"
    assert CANONICAL_HINTS["not too expensive"] == "p50"
    assert CANONICAL_HINTS["within reason"] == "p75"


def test_hint_to_band_maps_to_percentile_upper_bound():
    lo, hi = hint_to_band("cheap", _mc())
    assert lo == Decimal("0") and hi == Decimal("35.00")
    lo, hi = hint_to_band("not too expensive", _mc())
    assert hi == Decimal("60.00")
    lo, hi = hint_to_band("within reason", _mc())
    assert hi == Decimal("95.00")


def test_overspend_ceiling_is_min_p90_and_1_5x_p50():
    # 1.5 * 60 = 90 < 140 => 90.
    assert overspend_ceiling(_mc()) == Decimal("90.00")


def test_unknown_hint_raises():
    with pytest.raises(KeyError):
        hint_to_band("super premium luxury", _mc())


def test_tier_c_eligible_when_gray_zone_nonempty():
    # gray zone is band.hi < spend <= overspend_ceiling; non-empty iff hi < ceiling.
    assert is_tier_c_eligible("cheap", _mc()) is True          # 35 < 90
    assert is_tier_c_eligible("within reason", _mc()) is False  # 95 > 90 => no gray zone
