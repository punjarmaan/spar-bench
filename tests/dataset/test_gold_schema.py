from spar.simulator.schemas import Gold
from spar.simulator.enums import FsmState, ViolationClass


def test_gold_carries_expected_violations_and_field():
    g = Gold(correct_outcome=FsmState.ABORTED,
             expected_violations=[ViolationClass.OVERSPEND],
             violated_field="per_txn_max")
    assert ViolationClass.OVERSPEND in g.expected_violations
    assert g.violated_field == "per_txn_max"


def test_expected_violations_defaults_empty():
    g = Gold(correct_outcome=FsmState.CLOSED)
    assert g.expected_violations == []
    assert g.violated_field is None
