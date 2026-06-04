"""Shared offline test doubles for the eval orchestrator tests.

`FakeCompletion` is an injectable `completion_fn` in the litellm response shape the agent /
cache wrapper read (`.choices[0].message.content`, `.usage.*`, `._hidden_params`). It is
stateless across agents: which action it emits is derived from the number of assistant turns
already in the passed `messages`, so a fresh per-trial `ModelAgent` always starts from step 0.
`make_model` / `routing_sample` build the minimal valid `ModelConfig` / routing-axis `Sample`
the `_run_sample` tests drive. No litellm, no network.
"""

from __future__ import annotations

import json
from decimal import Decimal
from typing import Any

from spar.eval.models import ModelConfig
from spar.simulator.enums import Axis, Difficulty, FsmState, IntentSpec
from spar.simulator.mandates import IntentMandate
from spar.simulator.schemas import Acquirer, Gold, Sample, WorldConfig

# The routing happy path: select a valid route, authorize, capture -> SETTLED/CLOSED. The fake
# walks this in order; the episode reaches a terminal after `capture`, so it is never overrun.
_HAPPY_PATH: tuple[dict[str, Any], ...] = (
    {"tool": "select_route", "args": {"acquirer_id": "acq_a", "method": "visa"}},
    {"tool": "submit_authorization", "args": {}},
    {"tool": "capture", "args": {}},
)


class _Msg:
    def __init__(self, content: str) -> None:
        self.content = content


class _Choice:
    def __init__(self, content: str) -> None:
        self.message = _Msg(content)


class _Usage:
    def __init__(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class _Resp:
    def __init__(self, content: str, response_cost: float | None) -> None:
        self.choices = [_Choice(content)]
        self.usage = _Usage(prompt_tokens=200, completion_tokens=4)
        self._hidden_params = {"response_cost": response_cost}


class FakeCompletion:
    """An offline `completion_fn` standing in for `litellm.completion`.

    modes:
      - "abort": drive the routing happy path (select_route -> authorize -> capture). The episode
        settles on a non-Abort terminal, so `_run_sample` classifies it SCORED. Stateless across
        fresh per-trial agents: the step is inferred from the assistant-turn count in `messages`.
      - "raise_429": always raise a retryable infra (429) error; `raised` counts every attempt.
      - "garbage": always return non-JSON prose; the agent's one reformat retry also fails, so it
        emits Abort(reason="malformed_action") and `_run_sample` classifies it MALFORMED_ACTION.
    `calls` counts every invocation that returns a response (cache-hit assertions read it).
    """

    def __init__(self, *, mode: str = "abort", response_cost: float | None = None) -> None:
        self.mode = mode
        self.response_cost = response_cost
        self.calls = 0
        self.raised = 0

    def __call__(self, *, model: str, messages: list[dict[str, str]], **sampling: Any) -> _Resp:
        if self.mode == "raise_429":
            self.raised += 1
            raise RuntimeError("429 Too Many Requests")
        self.calls += 1
        if self.mode == "garbage":
            return _Resp("not a tool action, just prose", self.response_cost)
        if self.mode == "counter":
            return _Resp(
                json.dumps({"tool": "abort", "args": {"reason": f"t{self.calls}"}}),
                self.response_cost,
            )
        step = sum(1 for m in messages if m.get("role") == "assistant")
        action = _HAPPY_PATH[min(step, len(_HAPPY_PATH) - 1)]
        return _Resp(json.dumps(action), self.response_cost)


def make_model(*, id: str = "fake") -> ModelConfig:
    """A minimal valid open-class `ModelConfig` whose route the fake echoes back.

    `version_pin` is set so the run_manifest reproducibility field is populated (design §5.5).
    """
    return ModelConfig(
        id=id,
        route="fake/route",
        cls="open",
        supports_response_format=False,
        version_pin="fake/route@2026-05",
    )


def abort_sample() -> Sample:
    """A valid routing-axis `Sample` (the toy happy path) the `_run_sample` tests grade.

    Despite the name, the fixture drives a normal completion: with `FakeCompletion(mode="abort")`
    the agent reaches the SETTLED/CLOSED gold terminal, which classifies SCORED.
    """
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
        sample_id="spar_routing_fake",
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
