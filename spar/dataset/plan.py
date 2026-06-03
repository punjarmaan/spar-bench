"""The SINGLE split mechanism.

`plan_all(build_seed)` enumerates every PROCEDURAL sample for the public-procedural
splits (`lite`, `main`) and tags each with its split in ONE pass. Split membership
is a pure function of (sample_id, build_seed) by construction — there is NO second
hash-bucketing step layered on top, so the per-split / per-axis trap balance and the
model-graded cap the plan establishes are never scrambled. Diamond is the hand-authored
backbone, NOT enumerated here.
"""

from __future__ import annotations

from dataclasses import dataclass

from spar.simulator.enums import Axis, Difficulty, IntentSpec
from spar.simulator.rng import stable_hash

from spar.dataset.generator import GenSpec

TARGET_TRAP_FRACTION = 0.40

# The only procedurally-generated public splits. Diamond is hand-authored;
# Private is built separately from the full graded pool (build.py).
PROCEDURAL_SPLITS: tuple[str, ...] = ("lite", "main")

# Per-split sample count per axis (× 7 axes ≈ the target split size).
_PER_AXIS: dict[str, int] = {"lite": 9, "main": 86}

# Difficulty mix per split (cumulative thresholds on a [0,1) bucket).
_DIFF_MIX: dict[str, list[tuple[float, Difficulty]]] = {
    "lite": [(0.5, Difficulty.EASY), (0.9, Difficulty.MEDIUM), (1.0, Difficulty.HARD)],
    "main": [(0.34, Difficulty.EASY), (0.67, Difficulty.MEDIUM), (1.0, Difficulty.HARD)],
}


@dataclass(frozen=True)
class PlannedSpec:
    """A procedural GenSpec already bound to its split (the single mechanism)."""

    split: str
    spec: GenSpec


def _bucket(key: str) -> float:
    """Stable [0,1) bucket from a string key."""
    return (stable_hash(key) % 10_000) / 10_000.0


def _difficulty(split: str, key: str) -> Difficulty:
    b = _bucket(key + ":diff")
    for threshold, diff in _DIFF_MIX[split]:
        if b < threshold:
            return diff
    return Difficulty.HARD


def _intent_spec(axis: Axis, key: str) -> IntentSpec:
    # Semantic/underspecified live on consent_mandate. Keep the Tier-C-eligible
    # (semantic) share small so the model-graded cap never trips.
    if axis is not Axis.CONSENT_MANDATE:
        return IntentSpec.EXPLICIT
    b = _bucket(key + ":intent")
    if b < 0.10:
        return IntentSpec.SEMANTIC
    if b < 0.13:
        return IntentSpec.UNDERSPECIFIED
    return IntentSpec.EXPLICIT


def _plan_split(split: str, *, build_seed: int) -> list[GenSpec]:
    """Enumerate the balanced GenSpecs for one procedural split."""
    per_axis = _PER_AXIS[split]
    specs: list[GenSpec] = []
    for axis in Axis:
        n_traps = round(per_axis * TARGET_TRAP_FRACTION)
        explicit_trap_idx = 0  # counts traps assigned among explicit samples only
        trap_index = 0  # round-robin position among THIS axis's actual traps
        for i in range(per_axis):
            key = f"{split}:{build_seed}:{axis.value}:{i}"
            seed = stable_hash(key) % 1_000_000
            difficulty = _difficulty(split, key)
            intent = _intent_spec(axis, key)
            # Underspecified/semantic are gold-CLOSED/ESCALATED, not is_trap traps.
            this_trap_index: int | None = None
            if intent is not IntentSpec.EXPLICIT:
                is_trap = False
            else:
                # Assign traps among explicit samples to preserve the target fraction.
                is_trap = explicit_trap_idx < n_traps
                explicit_trap_idx += 1
                if is_trap:
                    # Round-robin catastrophic-class assignment over the axis's traps so every
                    # class clears the build coverage floor deterministically (a `seed % k` hash
                    # residue could cluster a class below the floor under some build seeds, since
                    # the build gate keys on intended_violation_class).
                    this_trap_index = trap_index
                    trap_index += 1
            specs.append(GenSpec(axis=axis, seed=seed, difficulty=difficulty,
                                 is_trap=is_trap, intent_spec=intent,
                                 trap_index=this_trap_index))
    return specs


def plan_all(*, build_seed: int) -> list[PlannedSpec]:
    """Every procedural sample, each bound to its split. The single mechanism:
    split membership is a pure function of (sample_id, build_seed) — no second pass."""
    planned: list[PlannedSpec] = []
    for split in PROCEDURAL_SPLITS:
        for spec in _plan_split(split, build_seed=build_seed):
            planned.append(PlannedSpec(split=split, spec=spec))
    return planned


def plan_samples(split: str, *, build_seed: int) -> list[GenSpec]:
    """A thin split-filtered view of `plan_all` (procedural splits only)."""
    if split not in PROCEDURAL_SPLITS:
        raise ValueError(f"unknown procedural split: {split!r}")
    return _plan_split(split, build_seed=build_seed)
