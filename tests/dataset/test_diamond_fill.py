from __future__ import annotations
from collections import Counter
from spar.dataset.generator import diamond_fill_cohort
from spar.simulator.enums import Axis


def test_fill_is_balanced_traps_with_own_ids():
    cohort = diamond_fill_cohort(per_axis=6, seed=1)
    assert all(s.is_trap for s in cohort)
    assert all("diamond_proc" in s.sample_id for s in cohort)
    per = Counter(s.axis for s in cohort)
    assert all(per[a.value] == 6 for a in Axis)          # balanced across all 7 axes
    assert len({s.sample_id for s in cohort}) == len(cohort)   # unique ids


def test_fill_is_deterministic():
    a = [s.sample_id for s in diamond_fill_cohort(per_axis=6, seed=1)]
    b = [s.sample_id for s in diamond_fill_cohort(per_axis=6, seed=1)]
    assert a == b


def test_fill_passes_diamond_gates():
    from spar.dataset.build import diamond_trippability_spotcheck, trap_mechanism_spotcheck
    cohort = diamond_fill_cohort(per_axis=6, seed=1)
    trap_mechanism_spotcheck(
        cohort, enforce=True, split="diamond",
        enforced_splits=frozenset({"diamond"}),
    )                                                    # must not raise
    diamond_trippability_spotcheck(cohort)               # must not raise


def test_fill_catastrophic_class_spread():
    """Each catastrophic axis spans >=2 distinct intended classes across its fill traps."""
    cohort = diamond_fill_cohort(per_axis=6, seed=1)
    by_axis: dict[str, set[str]] = {}
    for s in cohort:
        vc = s.gold.intended_violation_class
        if vc is None:
            continue
        by_axis.setdefault(s.axis.value, set()).add(vc.value)
    # consent_mandate, compliance_tax, post_purchase are the catastrophic axes
    assert len(by_axis["compliance_tax"]) >= 2
    assert len(by_axis["post_purchase"]) >= 2
    assert len(by_axis["consent_mandate"]) >= 2
