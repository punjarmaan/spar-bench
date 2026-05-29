"""Canonical enumerations and terminal-state sets for the Spar FSM and dataset."""

from __future__ import annotations

from enum import StrEnum


class Axis(StrEnum):
    ROUTING = "routing"
    DECLINE_RECOVERY = "decline_recovery"
    CONSENT_MANDATE = "consent_mandate"
    STALE_STATE = "stale_state"
    COMPLIANCE_TAX = "compliance_tax"
    FRAUD_REACTIVITY = "fraud_reactivity"
    POST_PURCHASE = "post_purchase"


class Difficulty(StrEnum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class IntentSpec(StrEnum):
    EXPLICIT = "explicit"
    SEMANTIC = "semantic"
    UNDERSPECIFIED = "underspecified"


class FsmState(StrEnum):
    CART = "CART"
    ROUTE_SELECTED = "ROUTE_SELECTED"
    AUTH_PENDING = "AUTH_PENDING"
    CHALLENGE = "CHALLENGE"
    APPROVED = "APPROVED"
    DECLINED = "DECLINED"
    CAPTURE_PENDING = "CAPTURE_PENDING"
    CAPTURED = "CAPTURED"
    SETTLED = "SETTLED"
    DISPUTED = "DISPUTED"
    CLOSED = "CLOSED"
    ABORTED = "ABORTED"
    ESCALATED = "ESCALATED"


class ToolStatus(StrEnum):
    OK = "ok"
    APPROVED = "approved"
    DECLINED = "declined"
    REQUIRES_ACTION = "requires_action"
    CAPTURE_INITIATED = "capture_initiated"
    CAPTURE_RESULT = "capture_result"
    SCOPE_VIOLATION = "scope_violation"
    ILLEGAL_ACTION = "illegal_action"
    ESCALATED = "escalated"
    ABORTED = "aborted"


# F3: the agent loop ends at an agent-terminal; CLOSED is resolved post-loop by the grade drain.
TERMINAL_AGENT: frozenset[FsmState] = frozenset(
    {FsmState.SETTLED, FsmState.ABORTED, FsmState.ESCALATED}
)
TERMINAL_GRADE: frozenset[FsmState] = frozenset(
    {FsmState.CLOSED, FsmState.ABORTED, FsmState.ESCALATED}
)
