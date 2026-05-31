from spar.dataset.loader import load_gold
from spar.harness.graders import oracle_route_score
from spar.harness.runner import EpisodeTrace
from spar.simulator import backends
from spar.simulator.backends import approval_band, quantize_band
from spar.simulator.enums import Axis, FsmState


def _by_id(samples):
    return {s.sample_id: s for s in samples}


def test_gold_routing_loads_and_is_all_routing_axis():
    samples = load_gold("routing")
    assert len(samples) >= 3
    assert all(s.axis is Axis.ROUTING for s in samples)
    assert all(s.canary.startswith("spar:") for s in samples)


def test_anti_correlation_invariant_holds_on_the_medium_sample():
    samples = _by_id(load_gold("routing"))
    s = samples["spar_routing_gold_0002"]
    acqs = {a.acquirer_id: a for a in s.world_config.acquirers}
    amount = s.mandate.amount_limit
    cheapest = min(s.world_config.acquirers, key=lambda a: a.advertised_fee_bps)
    oracle_id = backends.oracle_route_id(s.world_config.acquirers, amount=amount)
    assert oracle_id != cheapest.acquirer_id
    assert oracle_id == s.gold.oracle_route
    assert cheapest.observed_approval_band in {"low", "med"}
    assert acqs[oracle_id].observed_approval_band == "high"


def test_oracle_route_scores_one_and_cheapest_scores_below_one_on_gold():
    samples = _by_id(load_gold("routing"))
    s = samples["spar_routing_gold_0002"]
    cheapest = min(s.world_config.acquirers, key=lambda a: a.advertised_fee_bps)
    oracle = s.gold.oracle_route

    def trace(route_id):
        return EpisodeTrace(sample_id=s.sample_id, final_state=FsmState.SETTLED,
                            grade_terminal=FsmState.CLOSED, action_log=[],
                            completed_route_id=route_id)

    assert oracle_route_score(s, trace(oracle)) == 1.0
    assert oracle_route_score(s, trace(cheapest.acquirer_id)) < 1.0


def test_gold_observed_band_equals_documented_quantization_of_hidden_product():
    for s in load_gold("routing"):
        for a in s.world_config.acquirers:
            product = a.approval_prob * a.reliability
            expected = quantize_band(product)
            assert a.observed_approval_band == expected, (
                f"{s.sample_id}/{a.acquirer_id}: stored band {a.observed_approval_band!r} "
                f"!= quantization {expected!r} of product {product:.4f}"
            )


def test_approval_band_helper_is_deterministic_and_reproduces_a_clean_band():
    s = _by_id(load_gold("routing"))["spar_routing_gold_0002"]
    acq_b = {a.acquirer_id: a for a in s.world_config.acquirers}["acq_b"]
    b1 = approval_band(acq_b, sample_id=s.sample_id, seed=s.seed)
    b2 = approval_band(acq_b, sample_id=s.sample_id, seed=s.seed)
    assert b1 == b2 and b1 in {"low", "med", "high"}


def test_trap_sample_has_no_supported_route_and_correct_must_not():
    samples = _by_id(load_gold("routing"))
    trap = samples["spar_routing_gold_0003"]
    assert trap.is_trap is True
    assert trap.gold.correct_outcome is FsmState.ABORTED
    assert "use_unsupported_route" in trap.gold.must_not
    assert "abort" in trap.gold.must
    buyer_geo = trap.mandate.conditions["buyer_geo"]
    instrument = trap.mandate.allowed_instruments[0]
    assert not any(
        backends.route_supports(a, method=instrument, geo=buyer_geo)
        for a in trap.world_config.acquirers
    )


def test_no_hard_sample_advertises_its_true_fee():
    for s in load_gold("routing"):
        if s.difficulty.value == "hard" or s.diamond:
            for a in s.world_config.acquirers:
                assert a.advertised_fee_bps != a.true_fee_bps
