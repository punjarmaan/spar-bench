from decimal import Decimal

from spar.dataset.surface import draw_surface


def test_same_inputs_yield_identical_surface():
    a = draw_surface("spar_routing_0001", seed=42, n_acquirers=3)
    b = draw_surface("spar_routing_0001", seed=42, n_acquirers=3)
    assert a == b


def test_different_seed_changes_surface_details():
    a = draw_surface("spar_routing_0001", seed=42, n_acquirers=3)
    b = draw_surface("spar_routing_0001", seed=43, n_acquirers=3)
    assert a.merchant != b.merchant or a.amount != b.amount


def test_amount_is_decimal_within_band_and_geo_valid():
    s = draw_surface("spar_routing_0001", seed=42, n_acquirers=3)
    assert isinstance(s.amount, Decimal)
    assert Decimal("10.00") <= s.amount <= Decimal("400.00")
    assert s.buyer_geo in {"US", "CA", "GB", "DE", "FR"}


def test_acquirer_ids_are_distinct_and_count_matches():
    s = draw_surface("spar_routing_0001", seed=42, n_acquirers=4)
    assert len(s.acquirer_ids) == 4
    assert len(set(s.acquirer_ids)) == 4
