"""Typed difficulty knobs.

Each knob's default direction is monotone easy -> ridiculously-hard. `knobs_for`
picks the rung values for one (axis, difficulty, is_trap). Knobs are *structure*;
surface randomization happens in generator.py and never touches these values.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from spar.simulator.enums import Axis, Difficulty


class DeclineClass(StrEnum):
    SOFT = "soft"
    AMBIGUOUS = "ambiguous"
    HARD = "hard"
    COMPOUND = "compound"


class DifficultyKnobs(BaseModel):
    """The structural difficulty dials for one sample."""

    model_config = ConfigDict(extra="forbid")

    n_acquirers: int = Field(ge=1, le=5)
    decline_class: DeclineClass
    fraud_sensitivity: float = Field(ge=0.0, le=1.0)
    n_interacting_traps: int = Field(ge=0, le=3)
    step_up_prob: float = Field(ge=0.0, le=1.0)
    async_capture: bool
    delayed_dispute: bool
    signal_conflict: int = Field(ge=0, le=3)


def _base(difficulty: Difficulty) -> DifficultyKnobs:
    """Axis-agnostic rung baseline; axes override the knobs they actually vary."""
    if difficulty is Difficulty.EASY:
        return DifficultyKnobs(
            n_acquirers=1, decline_class=DeclineClass.SOFT, fraud_sensitivity=0.1,
            n_interacting_traps=0, step_up_prob=0.0, async_capture=False,
            delayed_dispute=False, signal_conflict=0,
        )
    if difficulty is Difficulty.MEDIUM:
        return DifficultyKnobs(
            n_acquirers=3, decline_class=DeclineClass.AMBIGUOUS, fraud_sensitivity=0.4,
            n_interacting_traps=1, step_up_prob=0.3, async_capture=False,
            delayed_dispute=False, signal_conflict=1,
        )
    return DifficultyKnobs(
        n_acquirers=5, decline_class=DeclineClass.HARD, fraud_sensitivity=0.7,
        n_interacting_traps=2, step_up_prob=0.6, async_capture=False,
        delayed_dispute=False, signal_conflict=3,
    )


def knobs_for(axis: Axis, difficulty: Difficulty, *, is_trap: bool) -> DifficultyKnobs:
    """Rung values for one sample, per the axis's difficulty ladder."""
    k = _base(difficulty)
    if axis is Axis.DECLINE_RECOVERY and difficulty is Difficulty.HARD:
        k.decline_class = DeclineClass.COMPOUND if is_trap else DeclineClass.HARD
    if axis is Axis.POST_PURCHASE:
        # async on hard; delayed dispute on diamond-hard traps.
        k.async_capture = difficulty in {Difficulty.MEDIUM, Difficulty.HARD}
        k.delayed_dispute = difficulty is Difficulty.HARD and is_trap
    if axis is Axis.FRAUD_REACTIVITY:
        k.step_up_prob = {Difficulty.EASY: 0.0, Difficulty.MEDIUM: 0.5, Difficulty.HARD: 0.8}[difficulty]
    if axis is Axis.STALE_STATE and difficulty is Difficulty.HARD and is_trap:
        k.delayed_dispute = True  # settle-time FX trap risks a delayed dispute
    return k
