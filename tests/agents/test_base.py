from decimal import Decimal

from spar.agents.base import Agent, AbortAgent
from spar.simulator.contract import Observation, ObsCart, ObsContext


def _obs() -> Observation:
    from spar.simulator.mandates import IntentMandate
    return Observation(
        mandate=IntentMandate(
            goal="buy widget", amount_limit=Decimal("100"), currency="USD",
            human_present=True, conditions={},
                per_txn_max=Decimal("200"), daily_remaining=Decimal("350"),
                merchant_constraint=["acme"], mcc_constraint=None,
                allowed_instruments=["visa"], session_ttl_steps=20, single_use_or_recurring="single_use", time_window=None,
        ),
        cart=ObsCart(line_items=[], subtotal=Decimal("10")),
        methods=[],
        context=ObsContext(buyer_geo="US", elapsed_steps=0),
    )


def test_abort_agent_satisfies_protocol_and_aborts():
    agent: Agent = AbortAgent()
    action = agent.act(_obs())
    assert action.tool == "abort"
