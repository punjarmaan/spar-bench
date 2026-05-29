from decimal import Decimal

import pytest

from spar.simulator.backends import (
    RouteOutcome,
    approval_band,
    quantize_band,
    find_acquirer,
    route_supports,
    resolve_authorization,
    expected_retry_cost_for,
    expected_dispute_cost_for,
    expected_value,
    enumerate_evs,
    oracle_route_id,
)
from spar.simulator.schemas import Acquirer


def _acq(
    acquirer_id: str,
    *,
    methods: list[str],
    geos: list[str],
    advertised_fee_bps: int,
    band: str,
    true_fee_bps: int,
    approval_prob: float,
    reliability: float,
) -> Acquirer:
    return Acquirer(
        acquirer_id=acquirer_id,
        methods=methods,
        supported_geos=geos,
        advertised_fee_bps=advertised_fee_bps,
        observed_approval_band=band,
        true_fee_bps=true_fee_bps,
        approval_prob=approval_prob,
        reliability=reliability,
    )


def _three_acquirers() -> list[Acquirer]:
    return [
        _acq("acq_a", methods=["visa", "mc"], geos=["US"], advertised_fee_bps=180,
             band="low", true_fee_bps=310, approval_prob=0.71, reliability=0.95),
        _acq("acq_b", methods=["visa", "mc"], geos=["US"], advertised_fee_bps=245,
             band="high", true_fee_bps=245, approval_prob=0.93, reliability=0.97),
        _acq("acq_c", methods=["visa"], geos=["US", "CA"], advertised_fee_bps=220,
             band="med", true_fee_bps=260, approval_prob=0.88, reliability=0.90),
    ]


def test_find_acquirer_returns_the_matching_object_or_none():
    acqs = _three_acquirers()
    assert find_acquirer(acqs, "acq_b").acquirer_id == "acq_b"
    assert find_acquirer(acqs, "missing") is None


def test_route_supports_checks_both_method_and_geo():
    acqs = _three_acquirers()
    assert route_supports(acqs[2], method="visa", geo="CA") is True
    assert route_supports(acqs[2], method="mc", geo="CA") is False
    assert route_supports(acqs[0], method="visa", geo="CA") is False


def test_expected_value_matches_hand_computed_value():
    acq_b = _three_acquirers()[1]
    ev = expected_value(acq_b, amount=Decimal("100"))
    assert ev == Decimal("90.7215")


def test_expected_value_subtracts_retry_and_dispute_costs():
    acq_b = _three_acquirers()[1]
    ev = expected_value(
        acq_b, amount=Decimal("100"),
        expected_retry_cost=Decimal("0.30"),
        expected_dispute_cost=Decimal("1.00"),
    )
    assert ev == Decimal("89.4215")


def test_expected_retry_cost_is_per_route_from_reliability_miss():
    acq_a = _three_acquirers()[0]
    cost = expected_retry_cost_for(acq_a, retry_penalty=Decimal("0.30"))
    assert cost == Decimal("0.0150")


def test_expected_dispute_cost_is_per_route_from_approval_and_dispute_prob():
    acq_b = _three_acquirers()[1]
    cost = expected_dispute_cost_for(
        acq_b, dispute_prob=Decimal("0.02"), dispute_penalty=Decimal("1.00")
    )
    assert cost == Decimal("0.0186")


def test_quantize_band_is_the_documented_threshold_quantization():
    acqs = _three_acquirers()
    assert quantize_band(acqs[0].approval_prob * acqs[0].reliability) == "low"
    assert quantize_band(acqs[1].approval_prob * acqs[1].reliability) == "high"
    assert quantize_band(acqs[2].approval_prob * acqs[2].reliability) == "med"
    assert quantize_band(0.749999) == "low"
    assert quantize_band(0.75) == "med"
    assert quantize_band(0.849999) == "med"
    assert quantize_band(0.85) == "high"


def test_approval_band_is_deterministic_and_within_one_level_of_the_clean_band():
    order = {"low": 0, "med": 1, "high": 2}
    for acq in _three_acquirers():
        clean = quantize_band(acq.approval_prob * acq.reliability)
        b1 = approval_band(acq, sample_id="spar_routing_x", seed=7)
        b2 = approval_band(acq, sample_id="spar_routing_x", seed=7)
        assert b1 == b2 and b1 in {"low", "med", "high"}
        assert abs(order[b1] - order[clean]) <= 1


def test_oracle_route_is_the_max_ev_route_with_per_route_costs():
    acqs = _three_acquirers()
    evs = enumerate_evs(acqs, amount=Decimal("100"))
    assert evs["acq_a"] == Decimal("68.7698")
    assert evs["acq_b"] == Decimal("90.6939")
    assert evs["acq_c"] == Decimal("85.6644")
    assert oracle_route_id(acqs, amount=Decimal("100")) == "acq_b"


def test_per_route_costs_can_flip_the_oracle_on_a_failover_sample():
    flaky_cheap = _acq("acq_flaky", methods=["visa"], geos=["US"], advertised_fee_bps=150,
                       band="med", true_fee_bps=150, approval_prob=0.99, reliability=0.40)
    reliable = _acq("acq_solid", methods=["visa"], geos=["US"], advertised_fee_bps=200,
                    band="high", true_fee_bps=200, approval_prob=0.97, reliability=0.99)
    acqs = [flaky_cheap, reliable]
    gross_only = enumerate_evs(
        acqs, amount=Decimal("100"), retry_penalty=Decimal("0"),
        dispute_prob=Decimal("0"), dispute_penalty=Decimal("0"),
    )
    assert gross_only["acq_flaky"] > gross_only["acq_solid"]
    assert oracle_route_id(
        acqs, amount=Decimal("100"), retry_penalty=Decimal("10.00"),
    ) == "acq_solid"


def test_resolve_authorization_is_deterministic_per_attempt_ordinal():
    acq_b = _three_acquirers()[1]
    o1 = resolve_authorization(
        acq_b, sample_id="spar_routing_x", seed=7, trial_index=0, attempt_ordinal=0,
    )
    o2 = resolve_authorization(
        acq_b, sample_id="spar_routing_x", seed=7, trial_index=0, attempt_ordinal=0,
    )
    assert isinstance(o1, RouteOutcome)
    assert o1 == o2


def test_resolve_authorization_draw_is_keyed_on_attempt_ordinal_not_elapsed_steps():
    flaky = _acq("acq_flaky", methods=["visa"], geos=["US"], advertised_fee_bps=200,
                 band="med", true_fee_bps=200, approval_prob=0.5, reliability=0.5)
    a = resolve_authorization(flaky, sample_id="s", seed=2, trial_index=0, attempt_ordinal=0)
    b = resolve_authorization(flaky, sample_id="s", seed=2, trial_index=0, attempt_ordinal=0)
    assert a == b
    c = resolve_authorization(flaky, sample_id="s", seed=2, trial_index=0, attempt_ordinal=1)
    assert isinstance(c, RouteOutcome)


def test_resolve_authorization_approves_a_certain_route_and_declines_a_dead_one():
    certain = _acq("acq_sure", methods=["visa"], geos=["US"], advertised_fee_bps=200,
                   band="high", true_fee_bps=200, approval_prob=1.0, reliability=1.0)
    dead = _acq("acq_dead", methods=["visa"], geos=["US"], advertised_fee_bps=200,
                band="low", true_fee_bps=200, approval_prob=0.0, reliability=1.0)
    ok = resolve_authorization(certain, sample_id="s", seed=1, trial_index=0, attempt_ordinal=0)
    bad = resolve_authorization(dead, sample_id="s", seed=1, trial_index=0, attempt_ordinal=0)
    assert ok.approved is True and ok.reason_code is None
    assert bad.approved is False and bad.reason_code == "05"


def test_resolve_authorization_reliability_miss_is_issuer_unavailable():
    unreliable = _acq("acq_flaky", methods=["visa"], geos=["US"], advertised_fee_bps=200,
                      band="med", true_fee_bps=200, approval_prob=1.0, reliability=0.0)
    out = resolve_authorization(unreliable, sample_id="s", seed=1, trial_index=0, attempt_ordinal=0)
    assert out.approved is False and out.reason_code == "91"


def test_expected_value_rejects_float_amount():
    acq_b = _three_acquirers()[1]
    with pytest.raises(TypeError):
        expected_value(acq_b, amount=100.0)  # type: ignore[arg-type]
