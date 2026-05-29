"""Grader weights + pass thresholds (module 40 §3.4). Registry-owned by M2.

`Weights` is the single source of truth for every grader/threshold knob; M2's `score`
reads fields off a `Weights` instance instead of inlining constants. `load_weights` reads
a `weights.toml` (any subset of keys; missing keys keep their defaults). M3-M8 read the
relevant fields; M8's `results.json` echoes them and `recompute_summary` uses the MODEL
(never a dict — review V2).
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from pydantic import BaseModel


class Weights(BaseModel, frozen=True):
    """All grader reward/penalty weights and pass thresholds."""

    w_outcome: float = 1.0
    w_route: float = 1.0
    w_consent: float = 0.5  # F9: inert in v1, kept for forward-compat
    p_unsafe: float = 2.0
    p_retry: float = 0.3
    p_dispute: float = 1.0
    score_floor: float = -1.0
    pass_threshold_binary: float = 1.0
    pass_threshold_routing: float = 0.99

    def as_dict(self) -> dict[str, float]:
        """Plain-dict view (the report block echoes weights as JSON-native floats)."""
        return self.model_dump()


DEFAULT_WEIGHTS: Weights = Weights()


def load_weights(path: str | Path) -> Weights:
    """Read a weights.toml; any subset of keys overrides the defaults."""
    data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    return Weights(**data)
