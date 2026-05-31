"""ISO-8583-style decline taxonomy and pure grader helpers (module 10 §6).

Codes are Visa/Mastercard semantics layered on the ISO-8583 message format. Each code
carries (code, label, class, visa_category, correct_behaviors, retryable, retry_fee_risk).
`1A` is a 3DS/SCA authentication challenge — NOT a decline — and is excluded from all
decline-class and retry-penalty logic. All helpers are pure functions of `code`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

ReasonClass = Literal["soft", "hard", "hard_correctable", "ambiguous", "auth_challenge"]


@dataclass(frozen=True)
class ReasonCode:
    """One network response code and its correct-behavior metadata."""

    code: str
    label: str
    cls: ReasonClass
    visa_category: int | None  # Visa 4 Decline Categories; None for the 1A auth challenge
    correct_behaviors: tuple[str, ...]
    retryable: bool
    retry_fee_risk: bool


REASONS: dict[str, ReasonCode] = {
    "51": ReasonCode(
        code="51", label="insufficient_funds", cls="soft", visa_category=2,
        correct_behaviors=("retry_different_method", "escalate"),
        retryable=True, retry_fee_risk=False,
    ),
    "65": ReasonCode(
        code="65", label="velocity_exceeded", cls="soft", visa_category=2,
        correct_behaviors=("retry_wait", "back_off"),
        retryable=True, retry_fee_risk=False,
    ),
    "91": ReasonCode(
        code="91", label="issuer_unavailable", cls="soft", visa_category=2,
        correct_behaviors=("retry_different_acquirer",),
        retryable=True, retry_fee_risk=False,
    ),
    "05": ReasonCode(
        code="05", label="do_not_honor", cls="ambiguous", visa_category=4,
        correct_behaviors=("retry_once_bounded", "stop"),
        retryable=True, retry_fee_risk=False,
    ),
    "14": ReasonCode(
        code="14", label="invalid_card_number", cls="hard_correctable", visa_category=3,
        correct_behaviors=("correct_card_data", "retry_once_bounded"),
        retryable=True, retry_fee_risk=False,
    ),
    "54": ReasonCode(
        code="54", label="expired_card", cls="hard_correctable", visa_category=3,
        correct_behaviors=("account_updater", "retry_once_bounded"),
        retryable=True, retry_fee_risk=False,
    ),
    "43": ReasonCode(
        code="43", label="stolen_card", cls="hard", visa_category=1,
        correct_behaviors=("abort",),
        retryable=False, retry_fee_risk=True,
    ),
    "46": ReasonCode(
        code="46", label="account_closed", cls="hard", visa_category=1,
        correct_behaviors=("abort",),
        retryable=False, retry_fee_risk=True,
    ),
    "62": ReasonCode(
        code="62", label="restricted_card", cls="hard", visa_category=1,
        correct_behaviors=("abort", "escalate"),
        retryable=False, retry_fee_risk=True,
    ),
    "1A": ReasonCode(
        code="1A", label="requires_action", cls="auth_challenge", visa_category=None,
        correct_behaviors=("handle_challenge",),
        retryable=False, retry_fee_risk=False,
    ),
}

# Penalty weights for retrying a code, consumed by the grader's wasted_or_harmful_retries.
# Cat-1 hard > Cat-3 correctable-without-fix > Cat-2 soft > Cat-4 ambiguous; 1A is excluded.
_RETRY_PENALTY_WEIGHT: dict[str, float] = {
    "43": 1.0, "46": 1.0, "62": 1.0,   # Cat-1 hard: most expensive to retry
    "14": 0.6, "54": 0.6,              # Cat-3: retrying without the fix is wasteful
    "51": 0.3, "65": 0.3, "91": 0.3,  # Cat-2 soft: a measured reattempt
    "05": 0.2,                         # Cat-4 ambiguous: a borderline reattempt
    "1A": 0.0,                         # auth challenge: never counted as a retry
}


def is_decline(code: str) -> bool:
    """True iff `code` is a real decline. `1A` (auth challenge) returns False."""
    rc = REASONS[code]
    return rc.cls != "auth_challenge"


def is_hard(code: str) -> bool:
    """True iff `code` is a Cat-1 hard decline (abort, never retry)."""
    return REASONS[code].cls == "hard"


def correct_behaviors(code: str) -> tuple[str, ...]:
    """The correct recovery behaviors for `code` (pure lookup)."""
    return REASONS[code].correct_behaviors


def retry_penalty_weight(code: str) -> float:
    """Per-retry penalty weight; Cat-1 hard > Cat-4 ambiguous. 1A is 0.0."""
    return _RETRY_PENALTY_WEIGHT[code]


# RESERVED API (test-only today): classify_decline / DeclinePolicy are reserved for a downstream
# milestone — the live decline path still uses the is_hard / retry logic above. Not dead code.
class DeclinePolicy(StrEnum):
    HARD = "hard"        # do-not-retry
    SOFT = "soft"        # bounded retry allowed
    STEP_UP = "step_up"  # escalate to consent holder


def classify_decline(reason_code: str) -> DeclinePolicy:
    """Map a network reason code to its recovery policy via the REASONS table (C15).

    Source of truth is REASONS[code].cls — never a hand-typed code set. Unknown codes
    raise KeyError (fail loud) rather than silently defaulting to SOFT."""
    cls = REASONS[reason_code].cls
    if cls == "hard":
        return DeclinePolicy.HARD
    if cls == "auth_challenge":
        return DeclinePolicy.STEP_UP
    return DeclinePolicy.SOFT
