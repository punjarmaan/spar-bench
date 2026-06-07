"""Catastrophic traps stamp gold.intended_violation_class = the configured mechanic."""
from spar.dataset.generator import generate
from spar.dataset.plan import plan_all
from spar.simulator.enums import Axis

_CATASTROPHIC_AXES = {Axis.CONSENT_MANDATE, Axis.COMPLIANCE_TAX, Axis.POST_PURCHASE}
_VALID = {"overspend", "mandate_breach", "misdirected_funds", "wrong_currency",
          "duplicate_charge", "premature_capture", "missed_reversal_window"}


def test_catastrophic_traps_stamp_intended_class():
    seen = 0
    for planned in plan_all(build_seed=1):
        spec = planned.spec
        sample = generate(spec)
        ivc = sample.gold.intended_violation_class
        if ivc is not None:
            seen += 1
            assert sample.axis in _CATASTROPHIC_AXES
            assert sample.is_trap
            assert ivc.value in _VALID
    assert seen > 0  # at least some catastrophic traps got stamped
