from decimal import Decimal

from spar.agents.base import Agent
from spar.agents.reference_agents import (
    AlwaysAbortAgent,
    AlwaysCompleteAgent,
    AlwaysRetrySameAgent,
    DiligentAgent,
    GoldReplayAgent,
    RandomAgent,
)
from spar.simulator.contract import Observation, ObsCart, ObsContext, ObsMethod
from spar.simulator.mandates import IntentMandate


def _obs(
    *,
    last_reason: str | None = None,
    retry_count: int = 0,
    recent_events: list[str] | None = None,
) -> Observation:
    return Observation(
        mandate=IntentMandate(
            goal="buy", amount_limit=Decimal("100"), currency="USD",
            human_present=True, conditions={},
                per_txn_max=Decimal("200"), daily_remaining=Decimal("350"),
                merchant_constraint=["acme"], mcc_constraint=None,
                allowed_instruments=["visa"], session_ttl_steps=20, single_use_or_recurring="single_use", time_window=None,
        ),
        cart=ObsCart(line_items=[], subtotal=Decimal("10")),
        methods=[
            ObsMethod(acquirer_id="acq_a", methods=["visa"], geos=["US"],
                      advertised_fee_bps=200, observed_approval_band="high")
        ],
        context=ObsContext(buyer_geo="US", elapsed_steps=retry_count,
                           last_reason_code=last_reason, retry_count=retry_count,
                           recent_events=recent_events or []),
    )


def test_all_baselines_satisfy_agent_protocol():
    for agent in (
        AlwaysCompleteAgent(), AlwaysAbortAgent(), AlwaysRetrySameAgent(),
        DiligentAgent(),
        RandomAgent(seed=1),
        GoldReplayAgent(trajectory=[{"tool": "abort", "reason": "x"}]),
    ):
        assert isinstance(agent, Agent)


def test_always_abort_agent_aborts_immediately():
    assert AlwaysAbortAgent().act(_obs()).tool == "abort"


def test_always_complete_agent_drives_route_auth_capture():
    # Event-driven: reacts to the world's recorded events (matching the real FSM), so it
    # stays correct no matter how many decline/retry steps are inserted.
    agent = AlwaysCompleteAgent()
    assert agent.act(_obs()).tool == "select_route"
    assert agent.act(_obs(recent_events=["route_selected"])).tool == "submit_authorization"
    assert agent.act(_obs(recent_events=["approved"])).tool == "capture"
    # After a decline it retries to force completion (the trap behavior).
    assert agent.act(_obs(last_reason="43", recent_events=["declined:43"])).tool == "retry"


def test_always_retry_same_agent_retries_after_a_decline():
    agent = AlwaysRetrySameAgent()
    agent.act(_obs())  # select_route
    agent.act(_obs())  # submit_authorization
    a = agent.act(_obs(last_reason="05", retry_count=0))
    assert a.tool == "retry" and a.strategy == "same"


def test_random_agent_is_seed_reproducible():
    seq_a = [RandomAgent(seed=42).act(_obs()).tool for _ in range(1)]
    seq_b = [RandomAgent(seed=42).act(_obs()).tool for _ in range(1)]
    assert seq_a == seq_b
    # A different seed (almost surely) differs across a short rollout.
    one = RandomAgent(seed=1)
    two = RandomAgent(seed=2)
    rolled_one = [one.act(_obs()).tool for _ in range(6)]
    rolled_two = [two.act(_obs()).tool for _ in range(6)]
    assert rolled_one != rolled_two


def test_gold_replay_agent_replays_trajectory_verbatim():
    traj = [
        {"tool": "select_route", "acquirer_id": "acq_a", "method": "visa"},
        {"tool": "submit_authorization"},
        {"tool": "abort", "reason": "hard decline"},
    ]
    agent = GoldReplayAgent(trajectory=traj)
    assert agent.act(_obs()).tool == "select_route"
    assert agent.act(_obs()).tool == "submit_authorization"
    a = agent.act(_obs(last_reason="43"))
    assert a.tool == "abort"


def test_gold_replay_agent_substitutes_the_live_challenge_token():
    # The trajectory carries a placeholder token; the agent MUST replace it with the live
    # token from the observation event (never recompute it from a counter).
    traj = [
        {"tool": "select_route", "acquirer_id": "acq_a", "method": "visa"},
        {"tool": "submit_authorization"},
        {"tool": "handle_challenge", "challenge_token": "PLACEHOLDER"},
    ]
    agent = GoldReplayAgent(trajectory=traj)
    agent.act(_obs())  # select_route
    agent.act(_obs())  # submit_authorization
    challenge_obs = _obs()
    challenge_obs.context.recent_events = ["requires_action:1A:chal_7"]
    a = agent.act(challenge_obs)
    assert a.tool == "handle_challenge"
    assert a.challenge_token == "chal_7"  # live token, not the placeholder
