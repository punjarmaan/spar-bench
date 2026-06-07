from spar.simulator.enums import Axis, Difficulty, IntentSpec, FsmState, ToolStatus, TERMINAL_AGENT, TERMINAL_GRADE, ViolationClass  # noqa: F401
from spar.simulator.rng import SubStream


def test_axes_are_the_seven_capability_axes():
    assert {a.value for a in Axis} == {
        "routing", "decline_recovery", "consent_mandate", "stale_state",
        "compliance_tax", "fraud_reactivity", "post_purchase",
    }


def test_agent_terminal_excludes_closed_and_includes_settled():
    # SETTLED is agent-terminal; CLOSED is grade-terminal only.
    assert FsmState.SETTLED in TERMINAL_AGENT
    assert FsmState.CLOSED not in TERMINAL_AGENT
    assert FsmState.CLOSED in TERMINAL_GRADE
    assert TERMINAL_AGENT == {FsmState.SETTLED, FsmState.ABORTED, FsmState.ESCALATED}


def test_tool_status_has_requires_action_and_scope_violation():
    vals = {s.value for s in ToolStatus}
    assert "requires_action" in vals and "scope_violation" in vals and "declined" in vals


def test_violation_class_has_seven_v1_members():
    members = {v.value for v in ViolationClass}
    assert members == {
        "overspend", "misdirected_funds", "duplicate_charge", "wrong_currency",
        "premature_capture", "missed_reversal_window", "mandate_breach",
    }


def test_violation_class_severity_order_covers_every_member():
    # Append-only guard: a new class added to the enum but not to severity_order()
    # would be silently dropped by worst_class(). This catches that drift.
    assert set(ViolationClass.severity_order()) == set(ViolationClass)
    assert len(ViolationClass.severity_order()) == len(set(ViolationClass.severity_order()))


def test_violation_class_severity_order_is_irreversibility():
    # Highest severity first; index 0 is most severe.
    assert ViolationClass.severity_order() == [
        ViolationClass.MISDIRECTED_FUNDS,
        ViolationClass.DUPLICATE_CHARGE,
        ViolationClass.PREMATURE_CAPTURE,
        ViolationClass.OVERSPEND,
        ViolationClass.WRONG_CURRENCY,
        ViolationClass.MISSED_REVERSAL_WINDOW,
        ViolationClass.MANDATE_BREACH,
    ]


def test_lifecycle_states_added():
    assert {FsmState.VOIDED, FsmState.REFUNDED, FsmState.CHARGED_BACK} <= set(FsmState)


def test_substream_appends_are_stable():
    # Existing keys MUST keep their integer values (append-only contract).
    assert SubStream.DECLINE == 0 and SubStream.APPROVAL_BAND == 6
    assert SubStream.VOID == 7 and SubStream.REFUND == 8 and SubStream.CHARGEBACK == 9
