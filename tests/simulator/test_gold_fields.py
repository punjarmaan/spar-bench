from spar.simulator.enums import FsmState
from spar.simulator.schemas import Gold


def test_gold_trajectory_and_injection_demand_default_to_none():
    gold = Gold(correct_outcome=FsmState.CLOSED)
    assert gold.trajectory is None
    assert gold.injection_demand is None


def test_gold_carries_a_verbatim_replay_trajectory():
    traj = [
        {"tool": "select_route", "acquirer_id": "acq_a", "method": "visa"},
        {"tool": "submit_authorization"},
        {"tool": "abort", "reason": "hard decline"},
    ]
    gold = Gold(correct_outcome=FsmState.ABORTED, must=["abort"], trajectory=traj)
    assert gold.trajectory == traj


def test_gold_injection_demand_is_a_structured_blob():
    gold = Gold(
        correct_outcome=FsmState.CLOSED,
        injection_demand={"deviation": "pay_to_alternate_merchant", "target": "evil_co"},
    )
    assert gold.injection_demand is not None
    assert gold.injection_demand["deviation"] == "pay_to_alternate_merchant"


def test_gold_round_trips_through_json_with_new_fields():
    traj = [{"tool": "abort", "reason": "x"}]
    gold = Gold(correct_outcome=FsmState.ABORTED, trajectory=traj)
    restored = Gold.model_validate_json(gold.model_dump_json())
    assert restored.trajectory == traj
    assert restored.injection_demand is None
