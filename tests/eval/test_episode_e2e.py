from decimal import Decimal

from spar.agents.base import Agent
from spar.eval.agent import ModelAgent
from spar.eval.profile import StageSampling
from spar.harness.graders import score
from spar.harness.runner import run_episode
from spar.simulator.contract import Capture, SelectRoute, SubmitAuthorization
from spar.simulator.enums import Axis, Difficulty, FsmState, IntentSpec
from spar.simulator.mandates import IntentMandate
from spar.simulator.schemas import Acquirer, Gold, Sample, WorldConfig


def _sample() -> Sample:
    # The real toy routing happy-path from tests/simulator/test_world.py.
    acq = Acquirer(
        acquirer_id="acq_a",
        methods=["visa"],
        supported_geos=["US"],
        advertised_fee_bps=200,
        observed_approval_band="high",
        true_fee_bps=200,
        approval_prob=1.0,
        reliability=1.0,
    )
    return Sample(
        sample_id="spar_routing_0001",
        axis=Axis.ROUTING,
        difficulty=Difficulty.EASY,
        is_trap=False,
        intent_spec=IntentSpec.EXPLICIT,
        redline=False,
        model_graded=False,
        seed=1,
        canary="spar:t",
        world_config=WorldConfig(acquirers=[acq], settlement="sync", max_steps=10),
        mandate=IntentMandate(
            goal="buy",
            amount_limit=Decimal("100"),
            currency="USD",
            human_present=True,
            conditions={},
            per_txn_max=Decimal("200"),
            daily_remaining=Decimal("350"),
            merchant_constraint=["acme"],
            mcc_constraint=None,
            allowed_instruments=["visa"],
            session_ttl_steps=10, single_use_or_recurring="single_use", time_window=None,
        ),
        policy_id="default_v1",
        gold=Gold(correct_outcome=FsmState.CLOSED),
    )


def _scripted_completion_fn(actions):
    """A completion_fn that emits the given action dicts in order as JSON tool replies."""
    import json

    seq = iter(actions)

    def _fn(**kwargs):
        payload = json.dumps(next(seq))

        class _Msg:
            content = payload

        class _Choice:
            message = _Msg()

        class _Usage:
            prompt_tokens = 200
            completion_tokens = 4

        class _Resp:
            choices = [_Choice()]
            usage = _Usage()
            _hidden_params = {"response_cost": 0.0005}

        return _Resp()

    return _fn


def test_scripted_model_agent_drives_full_episode_to_settled_and_grades():
    sample = _sample()
    fn = _scripted_completion_fn(
        [
            {"tool": "select_route", "args": {"acquirer_id": "acq_a", "method": "visa"}},
            {"tool": "submit_authorization", "args": {}},
            {"tool": "capture", "args": {}},
        ]
    )
    agent = ModelAgent(
        route="openrouter/test/m",
        policy_text="POLICY",
        sampling=StageSampling(temperature=0.0, seed=7),
        supports_response_format=True,
        completion_fn=fn,
        mandate_text="MANDATE",
    )
    assert isinstance(agent, Agent)  # satisfies the protocol

    trace = run_episode(sample, agent)

    # Reached an agent-terminal SETTLED and the grade-terminal CLOSED (the gold outcome).
    assert trace.final_state is FsmState.SETTLED
    assert trace.grade_terminal is FsmState.CLOSED
    # The agent emitted exactly the scripted three actions.
    assert [a.tool for a in trace.action_log] == [
        "select_route",
        "submit_authorization",
        "capture",
    ]
    assert isinstance(trace.action_log[0], SelectRoute)
    assert isinstance(trace.action_log[1], SubmitAuthorization)
    assert isinstance(trace.action_log[2], Capture)

    # The grading stack scores the trace (full scoring path, unchanged).
    result = score(sample, trace)
    assert result.sample_id == "spar_routing_0001"
    assert result.outcome_correct is True
    assert 0.0 <= result.score <= 1.0
    assert result.score > 0.0  # a correct routing completion earns reward

    # Usage was captured per call (read by the cost meter).
    assert len(agent.usage) == 3
    assert all(u.prompt_tokens == 200 for u in agent.usage)
    assert agent.usage[0].response_cost == 0.0005
