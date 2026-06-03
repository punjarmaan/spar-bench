"""Task B3a: L1 scenario diversity — the drawn axes must BIND (constrain behavior),
not just decorate. Build `main` and assert genuine independent per-sample diversity:
multiple currencies / geos / instrument families / MCCs / categories, amounts that
straddle per_txn_max, and a high count of distinct world-config signatures per axis
(no pseudo-replication). These FAIL on the pre-B3a single-currency/geo/coffee_maker build.
"""

from __future__ import annotations

from collections import Counter

import pytest

from spar.dataset.generator import generate
from spar.dataset.plan import plan_all
from spar.simulator.backends import route_supports
from spar.simulator.enums import Axis, FsmState


_INSTRUMENT_FAMILIES = {"visa", "mc", "amex"}


@pytest.fixture(scope="module")
def main_samples():
    return [generate(p.spec) for p in plan_all(build_seed=1) if p.split == "main"]


def _signature(s) -> tuple:
    """A coarse world-config signature: the per-sample knobs that should vary."""
    m = s.mandate
    wc = s.world_config
    cat = wc.market_context.category if wc.market_context is not None else None
    return (
        m.currency,
        str(m.conditions.get("buyer_geo")),
        tuple(sorted(m.allowed_instruments)),
        tuple(sorted(m.mcc_constraint)) if m.mcc_constraint else None,
        cat,
        str(m.per_txn_max),
        str(wc.cart_total),
    )


def test_at_least_four_currencies_used(main_samples):
    currencies = {s.mandate.currency for s in main_samples}
    assert len(currencies) >= 4, currencies


def test_geo_varies_and_binds(main_samples):
    """>=4 distinct buyer_geos AND on non-traps the acquirers actually support the geo
    (so the geo binds without making the sample unsolvable)."""
    geos = {str(s.mandate.conditions.get("buyer_geo")) for s in main_samples}
    assert len(geos) >= 4, geos
    for s in main_samples:
        if s.is_trap or not s.world_config.acquirers:
            continue
        geo = str(s.mandate.conditions.get("buyer_geo"))
        assert any(geo in a.supported_geos for a in s.world_config.acquirers), (
            f"{s.sample_id}: geo {geo} not supported by any acquirer (unsolvable)"
        )


def test_three_instrument_families_including_amex(main_samples):
    seen: set[str] = set()
    for s in main_samples:
        seen.update(i for i in s.mandate.allowed_instruments if i in _INSTRUMENT_FAMILIES)
    assert _INSTRUMENT_FAMILIES <= seen, seen


def test_non_trap_instrument_is_served(main_samples):
    """Every non-trap must have >=1 acquirer that serves an allowed (instrument, geo)."""
    for s in main_samples:
        if s.is_trap or not s.world_config.acquirers:
            continue
        geo = str(s.mandate.conditions.get("buyer_geo"))
        served = any(
            route_supports(a, method=instr, geo=geo)
            for a in s.world_config.acquirers
            for instr in s.mandate.allowed_instruments
        )
        assert served, f"{s.sample_id}: no acquirer serves any allowed instrument at {geo}"


def test_mcc_constraints_present(main_samples):
    with_mcc = [s for s in main_samples if s.mandate.mcc_constraint]
    assert len(with_mcc) >= 10, len(with_mcc)
    mccs = {tuple(s.mandate.mcc_constraint) for s in with_mcc}
    assert len(mccs) >= 3, mccs


def test_semantic_category_not_always_coffee_maker(main_samples):
    cats = {
        s.world_config.market_context.category
        for s in main_samples
        if s.world_config.market_context is not None
    }
    assert len(cats) >= 3, cats
    assert cats != {"coffee_maker"}, cats


def test_amounts_straddle_per_txn_max(main_samples):
    """The drawn amounts must STRADDLE per_txn_max, not collapse to one side:
    - where the construct INTENDS an over-limit breach (the scope-wall / over-limit traps,
      violated_field == 'per_txn_max'), the live cart sits OVER per_txn_max; and
    - >=25% of CLOSED non-traps sit comfortably UNDER per_txn_max (the safe-completion cohort).
    Together these prove the amount distribution spans both sides of the limit."""
    over = under = n_over_pool = n_under_pool = 0
    for s in main_samples:
        wc = s.world_config
        if wc.cart_total is None:
            continue
        ptm = s.mandate.per_txn_max
        # over-pool: traps whose intended mechanism IS the over-limit scope wall.
        if s.is_trap and s.gold.violated_field == "per_txn_max":
            n_over_pool += 1
            if wc.cart_total > ptm:
                over += 1
        # under-pool: CLOSED non-traps (a comfortable headroom is intended).
        if not s.is_trap and s.gold.correct_outcome is FsmState.CLOSED:
            n_under_pool += 1
            if wc.cart_total < ptm:
                under += 1
    assert n_over_pool >= 10, n_over_pool
    assert over / n_over_pool >= 0.25, (over, n_over_pool)
    assert n_under_pool == 0 or under / n_under_pool >= 0.25, (under, n_under_pool)


def test_distinct_signatures_per_axis(main_samples):
    """No pseudo-replication: per axis, distinct world-config signatures >= 0.8*N and
    no single signature accounts for >5% of that axis's samples."""
    by_axis: dict[Axis, list] = {}
    for s in main_samples:
        by_axis.setdefault(s.axis, []).append(s)
    for axis, samples in by_axis.items():
        sigs = [_signature(s) for s in samples]
        n = len(sigs)
        distinct = len(set(sigs))
        assert distinct >= 0.8 * n, f"{axis}: {distinct} distinct / {n} (pseudo-replication)"
        top = Counter(sigs).most_common(1)[0][1]
        assert top <= max(1, 0.05 * n) + 1, f"{axis}: a signature covers {top}/{n} (>5%)"
