from spar.dataset.acquirers import build_acquirers, oracle_route


def test_anti_correlation_cheapest_advertised_is_not_high_band():
    acqs = build_acquirers("spar_routing_0042", seed=7, n_acquirers=3)
    cheapest = min(acqs, key=lambda a: a.advertised_fee_bps)
    # F2: the cheapest *advertised* route must NOT be the high-band one.
    assert cheapest.observed_approval_band in {"low", "med"}


def test_oracle_route_maximizes_hidden_ev_not_cheapest_advertised():
    acqs = build_acquirers("spar_routing_0042", seed=7, n_acquirers=3)
    oracle = oracle_route(acqs)
    cheapest = min(acqs, key=lambda a: a.advertised_fee_bps)
    assert oracle.acquirer_id != cheapest.acquirer_id


def test_hard_samples_advertised_never_equals_true_fee():
    # axes/routing.md §8: in tension samples, advertised != true so reading
    # advertised as truth is not a shortcut.
    acqs = build_acquirers("spar_routing_0042", seed=7, n_acquirers=5)
    assert all(a.advertised_fee_bps != a.true_fee_bps for a in acqs)


def test_deterministic_per_seed():
    a = build_acquirers("spar_routing_0042", seed=7, n_acquirers=3)
    b = build_acquirers("spar_routing_0042", seed=7, n_acquirers=3)
    assert a == b
