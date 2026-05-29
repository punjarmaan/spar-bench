"""Evaluation profile (spec §5.3 / §6): per-STAGE sampling + the split×k plan.

Temperature is a property of the eval STAGE, not the model — pinned here and applied identically
to every model and scenario, so no model can be tuned. Two stages: `competence` (temp 0.0, Main
pass^1) and `reliability` (temp 0.7, Diamond pass^4); pass^k at temp 0 would make all k trajectories
identical on scripted worlds, measuring nothing.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict


class StageSampling(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    temperature: float
    top_p: float = 1.0
    max_tokens: int = 2048
    seed: int | None = None


class StagePlan(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    split: str
    k: int
    stage: Literal["competence", "reliability"]
    published: bool
