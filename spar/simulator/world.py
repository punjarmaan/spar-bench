"""Toy deterministic World for M1 (happy path + abort/escalate). Full FSM lands in M2."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from spar.simulator.contract import (
    Abort, Action, Capture, Observation, ObsCart, ObsContext, ObsMethod,
    RequestUserConfirmation, SelectRoute, SubmitAuthorization, ToolResponse,
)
from spar.simulator.enums import FsmState, ToolStatus, TERMINAL_AGENT
from spar.simulator.schemas import Sample


class World:
    def __init__(self, sample: Sample, *, trial_index: int = 0) -> None:
        self.sample = sample
        self.trial_index = trial_index
        self.state: FsmState = FsmState.CART
        self.elapsed_steps = 0
        self.action_log: list[Action] = []
        # Deferred-event queue (min-heap of (fire_at_step, seq, event)); the real async
        # capture + dispute draining lands in M5. M1 ships an empty queue + identity drain.
        self.deferred: list[tuple[int, int, Any]] = []
        # Hidden state the GRADER reads (never the agent). M5 populates incurred_dispute, etc.
        self.hidden_final_state: dict[str, Any] = {"incurred_dispute": False}

    def reset(self) -> Observation:
        self.state = FsmState.CART
        self.elapsed_steps = 0
        self.action_log = []
        self.deferred = []
        self.hidden_final_state = {"incurred_dispute": False}
        return self.observe()

    def is_agent_terminal(self) -> bool:
        return self.state in TERMINAL_AGENT

    def drain_deferred(self) -> FsmState:
        """Resolve the GRADE-terminal after the agent loop ends (review root-cause fix / F3).

        The agent loop stops at an agent-terminal (`SETTLED`/`ABORTED`/`ESCALATED`); grading
        is done on the grade-terminal this returns. In M1 the queue is empty, so this is the
        identity `SETTLED → CLOSED` (a settled purchase with no dispute closes cleanly);
        `ABORTED`/`ESCALATED` pass through unchanged. M5 replaces the body with the real drain
        (async capture results + `DISPUTED → CLOSED` at `settle+K`), setting
        `hidden_final_state["incurred_dispute"]`.
        """
        if self.state is FsmState.SETTLED:
            return FsmState.CLOSED
        return self.state

    def observe(self) -> Observation:
        wc = self.sample.world_config
        return Observation(
            mandate=self.sample.mandate,
            cart=ObsCart(line_items=[], subtotal=self.sample.mandate.price_ceiling or Decimal(0)),
            methods=[
                ObsMethod(acquirer_id=a.acquirer_id, methods=a.methods, geos=a.supported_geos,
                          advertised_fee_bps=a.advertised_fee_bps,
                          observed_approval_band=a.observed_approval_band)
                for a in wc.acquirers
            ],
            context=ObsContext(buyer_geo="US", elapsed_steps=self.elapsed_steps),
        )

    def step(self, action: Action) -> ToolResponse:
        self.action_log.append(action)
        self.elapsed_steps += 1
        match action:
            case Abort():
                self.state = FsmState.ABORTED
                return ToolResponse(status=ToolStatus.ABORTED)
            case RequestUserConfirmation():
                self.state = FsmState.ESCALATED
                return ToolResponse(status=ToolStatus.ESCALATED)
            case SelectRoute():
                self.state = FsmState.ROUTE_SELECTED
                return ToolResponse(status=ToolStatus.OK)
            case SubmitAuthorization() if self.state is FsmState.ROUTE_SELECTED:
                self.state = FsmState.APPROVED
                return ToolResponse(status=ToolStatus.APPROVED)
            case Capture() if self.state is FsmState.APPROVED:
                # M1 toy: sync settlement, capture always succeeds.
                self.state = FsmState.SETTLED
                return ToolResponse(status=ToolStatus.OK)
            case _:
                return ToolResponse(status=ToolStatus.ILLEGAL_ACTION,
                                    detail={"state": self.state.value})
