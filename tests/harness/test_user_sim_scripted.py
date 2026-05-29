from decimal import Decimal

from spar.harness.user_sim import (
    UserResponse, UserSimRequest, UserSim, ScriptedUserSim,
)


def _req(reason: str = "May I spend more than expected?") -> UserSimRequest:
    return UserSimRequest(goal="buy a not-too-expensive coffee maker", reason=reason)


def test_scripted_user_sim_returns_approved_bound():
    sim: UserSim = ScriptedUserSim(
        UserResponse(decision="approve_bound", bound=Decimal("50"), message="sure, under $50")
    )
    resp = sim.respond(_req())
    assert resp.decision == "approve_bound"
    assert resp.bound == Decimal("50")
    assert resp.message == "sure, under $50"


def test_scripted_user_sim_returns_denial():
    sim = ScriptedUserSim(UserResponse(decision="deny", message="no, do not buy that"))
    resp = sim.respond(_req())
    assert resp.decision == "deny"
    assert resp.bound is None


def test_scripted_user_sim_request_carries_goal_not_tool_calls():
    req = _req()
    assert req.goal.startswith("buy a")
    assert not hasattr(req, "action_log")


def test_user_response_bound_must_not_be_float():
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        UserResponse(decision="approve_bound", bound=50.0)


def test_scripted_user_sim_satisfies_protocol():
    sim = ScriptedUserSim(UserResponse(decision="approve"))
    assert isinstance(sim, UserSim)
