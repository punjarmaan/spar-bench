from spar.simulator.enums import Axis, Difficulty, IntentSpec, FsmState, ToolStatus, TERMINAL_AGENT, TERMINAL_GRADE  # noqa: F401


def test_axes_are_the_seven_capability_axes():
    assert {a.value for a in Axis} == {
        "routing", "decline_recovery", "consent_mandate", "stale_state",
        "compliance_tax", "fraud_reactivity", "post_purchase",
    }


def test_agent_terminal_excludes_closed_and_includes_settled():
    # F3: SETTLED is agent-terminal; CLOSED is grade-terminal only.
    assert FsmState.SETTLED in TERMINAL_AGENT
    assert FsmState.CLOSED not in TERMINAL_AGENT
    assert FsmState.CLOSED in TERMINAL_GRADE
    assert TERMINAL_AGENT == {FsmState.SETTLED, FsmState.ABORTED, FsmState.ESCALATED}


def test_tool_status_has_requires_action_and_scope_violation():
    vals = {s.value for s in ToolStatus}
    assert "requires_action" in vals and "scope_violation" in vals and "declined" in vals
