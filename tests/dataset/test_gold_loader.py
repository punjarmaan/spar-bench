from spar.dataset.loader import load_gold
from spar.simulator.enums import Axis


def test_load_gold_consent_mandate_returns_samples():
    samples = load_gold(Axis.CONSENT_MANDATE)
    assert len(samples) >= 12
    assert all(s.axis is Axis.CONSENT_MANDATE for s in samples)


def test_load_gold_unknown_axis_returns_empty():
    samples = load_gold(Axis.POST_PURCHASE)  # no gold file shipped yet (M5)
    assert samples == []


def test_consent_trap_subset_is_nonempty():
    traps = [s for s in load_gold(Axis.CONSENT_MANDATE) if s.is_trap]
    assert len(traps) >= 9
