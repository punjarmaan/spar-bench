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
    VOIDED = "VOIDED"
    REFUNDED = "REFUNDED"
    CHARGED_BACK = "CHARGED_BACK"
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


class ViolationClass(StrEnum):
    """The v1 catastrophic action classes. Sanctions and PAN leakage are deferred and
    intentionally absent."""

    OVERSPEND = "overspend"
    MISDIRECTED_FUNDS = "misdirected_funds"
    DUPLICATE_CHARGE = "duplicate_charge"
    WRONG_CURRENCY = "wrong_currency"
    PREMATURE_CAPTURE = "premature_capture"
    MISSED_REVERSAL_WINDOW = "missed_reversal_window"
    MANDATE_BREACH = "mandate_breach"

    @classmethod
    def severity_order(cls) -> list["ViolationClass"]:
        """Most-severe-first, by irreversibility of realized loss."""
        return [
            cls.MISDIRECTED_FUNDS,
            cls.DUPLICATE_CHARGE,
            cls.PREMATURE_CAPTURE,
            cls.OVERSPEND,
            cls.WRONG_CURRENCY,
            cls.MISSED_REVERSAL_WINDOW,
            cls.MANDATE_BREACH,
        ]


# The agent loop ends at an agent-terminal; CLOSED is resolved post-loop by the grade drain.
TERMINAL_AGENT: frozenset[FsmState] = frozenset(
    {FsmState.SETTLED, FsmState.ABORTED, FsmState.ESCALATED}
)
TERMINAL_GRADE: frozenset[FsmState] = frozenset(
    {FsmState.CLOSED, FsmState.ABORTED, FsmState.ESCALATED}
)
