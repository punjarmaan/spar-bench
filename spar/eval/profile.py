"""Evaluation profile: per-STAGE sampling + the split×k plan.

Temperature is a property of the eval STAGE, not the model — pinned here and applied identically
to every model and scenario, so no model can be tuned. Two stages: `competence` (temp 0.0, Main
pass^1) and `reliability` (temp 0.7, Redline pass^4); pass^k at temp 0 would make all k trajectories
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


class Profile(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    competence: StageSampling          # Main pass^1 + per-axis breakdown
    reliability: StageSampling         # Redline pass^4
    plan: list[StagePlan]              # ordered (split, k, stage, published) entries


def load_profile(path: str | Path) -> Profile:
    """Read a profile TOML. Stage tables may be nested under `[profile.*]` or top-level
    (`[competence]`, `[reliability]`); `[[plan]]` is always top-level."""
    data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    # Prefer the nested `[profile.*]` form; fall back to top-level `[competence]`/`[reliability]`.
    profile = data.get("profile") or data
    return Profile(
        competence=StageSampling.model_validate(profile["competence"]),
        reliability=StageSampling.model_validate(profile["reliability"]),
        plan=[StagePlan.model_validate(row) for row in data.get("plan", [])],
    )


# Profile "A": the published default.
DEFAULT_PROFILE: Profile = Profile(
    competence=StageSampling(temperature=0.0, top_p=1.0, max_tokens=2048, seed=7),
    reliability=StageSampling(temperature=0.7, top_p=1.0, max_tokens=2048, seed=7),
    plan=[
        StagePlan(split="main", k=1, stage="competence", published=True),
        StagePlan(split="redline", k=4, stage="reliability", published=True),
        StagePlan(split="lite", k=1, stage="competence", published=False),
    ],
)
