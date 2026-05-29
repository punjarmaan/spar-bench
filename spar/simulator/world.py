"""The real payment FSM (module 10 §2.1). Replaces the M1 toy World.

Owns hidden world state and all randomness; advances on agent actions, resolving auth
outcomes via the decline_plan resolver. Exposes redacted Observations and structured
ToolResponses. The deferred-event queue is a stub here (full impl is M5). Money is Decimal;
randomness flows through spar.simulator.rng only.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from spar.simulator import backends
from spar.simulator.contract import (
    Abort,
    Action,
    Capture,
    HandleChallenge,
    ModifyCart,
    Observation,
    ObsCart,
    ObsContext,
    ObsMethod,
    RequestUserConfirmation,
    Retry,
    SelectRoute,
    SubmitAuthorization,
    ToolResponse,
)
from spar.simulator.decline_plan import resolve_auth_outcome, resolve_challenge_outcome
from spar.simulator.enums import TERMINAL_AGENT, FsmState, ToolStatus
from spar.simulator.schemas import Sample


class World:
    """Deterministic payment-FSM world for one sample/trial."""

    def __init__(self, sample: Sample, *, trial_index: int = 0) -> None:
        self.sample = sample
        self.trial_index = trial_index
        self.state: FsmState = FsmState.CART
        self.elapsed_steps = 0
        self.action_log: list[Action] = []
        self._tool_responses: list[ToolResponse] = []
        self._selected_acquirer: str | None = None
        self._selected_method: str | None = None
        self._challenge_token: str | None = None
        # M3 routing: the acquirer captured on a settled purchase (read by the oracle grader).
        self.completed_route_id: str | None = None
        # G2: stable per-route authorization-attempt ordinal — keys every DECLINE/CHALLENGE
        # draw, NEVER elapsed_steps. Bumped once per auth/challenge resolution (decline_plan path).
        self._auth_attempt = 0
        # M3 routing: per-acquirer authorization-attempt ordinals for the backend auth path.
        self.auth_attempts: dict[str, int] = {}
        # F8: per-reason-code attempt history, total retry count, bounded recent events.
        self.attempt_counts: dict[str, int] = {}
        self.retry_count = 0
        self.recent_events: list[str] = []
        # Deferred-event queue (frozen M1 contract): min-heap of (fire_at_step, seq, event);
        # full async-capture + dispute draining lands in M5.
        self._deferred: list[tuple[int, int, dict[str, Any]]] = []
        self._deferred_seq = 0
        # Hidden state the GRADER reads (never the agent); frozen M1 shape.
        self.hidden_final_state: dict[str, Any] = {"incurred_dispute": False}

    # ---- lifecycle ----

    def reset(self) -> Observation:
        self.state = FsmState.CART
        self.elapsed_steps = 0
        self.action_log = []
        self._tool_responses = []
        self._selected_acquirer = None
        self._selected_method = None
        self._challenge_token = None
        self.completed_route_id = None
        self._auth_attempt = 0
        self.auth_attempts = {}
        self.attempt_counts = {}
        self.retry_count = 0
        self.recent_events = []
        self._deferred = []
        self._deferred_seq = 0
        self.hidden_final_state = {"incurred_dispute": False}
        return self.observe()

    def is_agent_terminal(self) -> bool:
        return self.state in TERMINAL_AGENT

    def _buyer_geo(self) -> str:
        return str(self.sample.mandate.conditions.get("buyer_geo", "US"))

    # ---- deferred-event queue (frozen M1 contract; M5 fleshes out the min-heap drain) ----

    def _schedule_event(self, fire_at_step: int, event: dict[str, Any]) -> None:
        self._deferred.append((fire_at_step, self._deferred_seq, event))
        self._deferred_seq += 1

    def drain_deferred(self) -> FsmState:
        """Resolve the GRADE-terminal after the agent loop ends (frozen M1 contract / F3).

        M2 schedules nothing, so this is the identity `SETTLED → CLOSED`; `ABORTED`/`ESCALATED`
        pass through unchanged. M5 replaces the body with the real async-capture + `DISPUTED →
        CLOSED` draining, setting `hidden_final_state["incurred_dispute"]`.
        """
        if self.state is FsmState.SETTLED:
            return FsmState.CLOSED
        return self.state

    # ---- observation ----

    def _record_event(self, event: str) -> None:
        self.recent_events.append(event)
        if len(self.recent_events) > 8:
            self.recent_events = self.recent_events[-8:]

    def observe(self) -> Observation:
        wc = self.sample.world_config
        last_event = self.recent_events[-1] if self.recent_events else None
        last_reason: str | None = None
        if last_event and last_event.startswith("declined:"):
            last_reason = last_event.split(":", 1)[1]
        subtotal = self.sample.mandate.price_ceiling or Decimal("0")
        return Observation(
            mandate=self.sample.mandate,
            cart=ObsCart(line_items=[], subtotal=subtotal),
            methods=[
                ObsMethod(
                    acquirer_id=a.acquirer_id, methods=a.methods, geos=a.supported_geos,
                    advertised_fee_bps=a.advertised_fee_bps,
                    observed_approval_band=a.observed_approval_band,
                )
                for a in wc.acquirers
            ],
            context=ObsContext(
                buyer_geo=self._buyer_geo(),
                elapsed_steps=self.elapsed_steps,
                last_event=last_event,
                last_reason_code=last_reason,
                attempt_counts=dict(self.attempt_counts),
                retry_count=self.retry_count,
                recent_events=list(self.recent_events),
            ),
        )

    # ---- transition ----

    def _resolve_auth(self) -> ToolResponse:
        plan = self.sample.world_config.decline_plan
        # Dispatch: a sample with a scripted/sampled `decline_plan.mode` uses the M2
        # decline-taxonomy resolver (decline_recovery axis); a sample WITHOUT a plan mode
        # resolves stochastically against the selected acquirer's hidden params via the
        # backend (routing axis). Both key their draw on a per-route attempt ordinal (G2).
        if not plan.get("mode"):
            return self._resolve_backend_auth()
        # G2: bump the stable attempt ordinal once per resolution; key the draw on it.
        self._auth_attempt += 1
        outcome = resolve_auth_outcome(
            plan,
            sample_id=self.sample.sample_id,
            seed=self.sample.seed,
            trial_index=self.trial_index,
            attempt=self._auth_attempt,
        )
        if outcome.kind == "challenge":
            self.state = FsmState.CHALLENGE
            self._challenge_token = f"chal_{self._auth_attempt}"
            # Embed the LIVE token in the event so GoldReplayAgent reads it verbatim
            # (never recomputes it from a counter — review item: avoids desync).
            self._record_event(f"requires_action:1A:{self._challenge_token}")
            return ToolResponse(
                status=ToolStatus.REQUIRES_ACTION, challenge_token=self._challenge_token
            )
        if outcome.kind == "decline":
            return self._decline(outcome.reason_code or "05")
        self.state = FsmState.APPROVED
        self._record_event("approved")
        return ToolResponse(status=ToolStatus.APPROVED)

    def _resolve_backend_auth(self) -> ToolResponse:
        # M3 routing: resolve against the SELECTED acquirer's hidden approval_prob/reliability
        # (backends.resolve_authorization). Keyed on the PER-ROUTE attempt ordinal (read before
        # incrementing) so unrelated/illegal steps never shift the pinned draw (G2).
        acq = backends.find_acquirer(
            self.sample.world_config.acquirers, self._selected_acquirer or ""
        )
        assert acq is not None  # ROUTE_SELECTED guarantees a valid prior selection
        ordinal = self.auth_attempts.get(acq.acquirer_id, 0)
        self.auth_attempts[acq.acquirer_id] = ordinal + 1
        outcome = backends.resolve_authorization(
            acq, sample_id=self.sample.sample_id, seed=self.sample.seed,
            trial_index=self.trial_index, attempt_ordinal=ordinal,
        )
        if outcome.approved:
            self.state = FsmState.APPROVED
            self._record_event("approved")
            return ToolResponse(status=ToolStatus.APPROVED)
        return self._decline(outcome.reason_code or "05")

    def _decline(self, code: str) -> ToolResponse:
        self.state = FsmState.DECLINED
        self.attempt_counts[code] = self.attempt_counts.get(code, 0) + 1
        self._record_event(f"declined:{code}")
        return ToolResponse(status=ToolStatus.DECLINED, reason_code=code)

    def _resolve_challenge(self) -> ToolResponse:
        # G2: a 3DS step-up resolves in its OWN cleared/failed space, NOT a re-rolled auth
        # categorical (which could self-loop another `challenge`). Keyed on the same attempt.
        plan = self.sample.world_config.decline_plan
        outcome = resolve_challenge_outcome(
            plan,
            sample_id=self.sample.sample_id,
            seed=self.sample.seed,
            trial_index=self.trial_index,
            attempt=self._auth_attempt,
        )
        if outcome.kind == "failed":
            return self._decline(outcome.reason_code or "05")
        self.state = FsmState.APPROVED
        self._record_event("approved")
        return ToolResponse(status=ToolStatus.APPROVED)

    def _illegal(self) -> ToolResponse:
        return ToolResponse(
            status=ToolStatus.ILLEGAL_ACTION, detail={"state": self.state.value}
        )

    def step(self, action: Action) -> ToolResponse:
        self.action_log.append(action)
        self.elapsed_steps += 1
        resp = self._transition(action)
        self._tool_responses.append(resp)
        return resp

    def _transition(self, action: Action) -> ToolResponse:
        if self.is_agent_terminal():
            # Terminal states are absorbing: no action may cause a transition out.
            return self._illegal()
        match action:
            case Abort():
                self.state = FsmState.ABORTED
                self._record_event("aborted")
                return ToolResponse(status=ToolStatus.ABORTED)
            case RequestUserConfirmation():
                # M2 ships no responder -> terminal handoff (F4 terminal mode).
                self.state = FsmState.ESCALATED
                self._record_event("escalated")
                return ToolResponse(status=ToolStatus.ESCALATED)
            case SelectRoute() if self.state in (
                FsmState.CART,
                FsmState.ROUTE_SELECTED,
                FsmState.DECLINED,
            ):
                acq = next(
                    (a for a in self.sample.world_config.acquirers
                     if a.acquirer_id == action.acquirer_id),
                    None,
                )
                if acq is None:
                    return ToolResponse(
                        status=ToolStatus.ILLEGAL_ACTION,
                        detail={"reason": "unknown_acquirer", "acquirer_id": action.acquirer_id},
                    )
                if action.method not in acq.methods:
                    return ToolResponse(
                        status=ToolStatus.ILLEGAL_ACTION, detail={"reason": "unsupported_method"}
                    )
                if self._buyer_geo() not in acq.supported_geos:
                    return ToolResponse(
                        status=ToolStatus.ILLEGAL_ACTION, detail={"reason": "unsupported_geo"}
                    )
                self._selected_acquirer = action.acquirer_id
                self._selected_method = action.method
                self.state = FsmState.ROUTE_SELECTED
                self._record_event("route_selected")
                return ToolResponse(status=ToolStatus.OK)
            case SubmitAuthorization() if self.state is FsmState.ROUTE_SELECTED:
                return self._resolve_auth()
            case HandleChallenge() if self.state is FsmState.CHALLENGE:
                if action.challenge_token != self._challenge_token:
                    return self._illegal()
                self._record_event("challenge_handled")
                return self._resolve_challenge()
            case Retry() if self.state is FsmState.DECLINED:
                self.retry_count += 1
                self._record_event(f"retry:{action.strategy}")
                if action.strategy == "different_acquirer":
                    # Failover: clear the selection so the agent must re-select a route.
                    self._selected_acquirer = None
                    self._selected_method = None
                    self.state = FsmState.CART
                else:  # same / different_method / wait: re-auth the same selected route
                    self.state = FsmState.ROUTE_SELECTED
                return ToolResponse(status=ToolStatus.OK, detail={"retry": action.strategy})
            case ModifyCart() if self.state in (FsmState.CART, FsmState.DECLINED):
                # Data correction is legal pre-cart AND after a correctable decline (14/54):
                # the agent fixes the PAN / refreshes via Account Updater, then retries.
                self._record_event("cart_modified")
                return ToolResponse(status=ToolStatus.OK)
            case Capture() if self.state is FsmState.APPROVED:
                # sync settlement: capture advances straight to SETTLED (agent-terminal, F3).
                self.completed_route_id = self._selected_acquirer  # M3: exposed to the grader
                self.state = FsmState.SETTLED
                self._record_event("settled")
                return ToolResponse(status=ToolStatus.OK)
            case _:
                return self._illegal()
