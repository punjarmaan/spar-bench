"""Full-set catastrophic detection: a tripped flag is found regardless of gold_expected (Issue 3)."""
from spar.harness.violations import detect_all_violations
from spar.simulator.enums import ViolationClass


def test_detects_any_tripped_flag_ignoring_gold_expected():
    hidden = {"wrong_currency": True, "exceeded_ceiling": False}
    found = detect_all_violations(hidden_final_state=hidden)
    assert ViolationClass.WRONG_CURRENCY in found
    assert ViolationClass.OVERSPEND not in found


def test_empty_when_no_flag_set():
    assert detect_all_violations(hidden_final_state={}) == set()
