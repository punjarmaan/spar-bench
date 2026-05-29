from spar.dataset.loader import load_gold
from spar.simulator.enums import Axis


def test_load_gold_by_str_and_enum_returns_validated_samples():
    by_str = load_gold("decline_recovery")
    by_enum = load_gold(Axis.DECLINE_RECOVERY)
    assert len(by_str) == 10
    assert len(by_enum) == 10
    assert by_str[0].axis is Axis.DECLINE_RECOVERY
    assert by_str == by_enum
