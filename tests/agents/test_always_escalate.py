"""AlwaysEscalateAgent: the do-nothing-but-escalate adversary (gaming-floor baseline)."""
from spar.agents.base import Agent
from spar.agents.reference_agents import AlwaysEscalateAgent
from spar.simulator.contract import RequestUserConfirmation


def test_always_escalate_first_action_is_request_user_confirmation():
    a = AlwaysEscalateAgent()
    assert isinstance(a, Agent)

    class _Obs:  # minimal duck-typed observation; the agent ignores its contents
        pass
    act = a.act(_Obs())
    assert isinstance(act, RequestUserConfirmation)
    assert act.tool == "request_user_confirmation"
    assert isinstance(a.act(_Obs()), RequestUserConfirmation)
