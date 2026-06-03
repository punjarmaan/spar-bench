"""Gold carries an explicit intended_violation_class for catastrophic traps (Issue 3)."""
from spar.simulator.enums import FsmState, ViolationClass
from spar.simulator.schemas import Gold


def test_gold_defaults_intended_class_none():
    g = Gold(correct_outcome=FsmState.CLOSED)
    assert g.intended_violation_class is None


def test_gold_accepts_intended_class():
    g = Gold(correct_outcome=FsmState.ESCALATED, intended_violation_class=ViolationClass.OVERSPEND)
    assert g.intended_violation_class is ViolationClass.OVERSPEND
    assert Gold.model_validate_json(g.model_dump_json()).intended_violation_class is ViolationClass.OVERSPEND
