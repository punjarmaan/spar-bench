"""Results filenames disambiguate by stage when a split repeats in the plan.

A profile that runs one split at two stages (e.g. lite competence k=1 + lite reliability k=4)
must NOT have the second stage silently overwrite the first's `lite.results.json`. A repeated
split becomes `<split>.<stage>.results.json`; a unique split keeps `<split>.results.json` so
consolidate.py still finds `main.results.json` / `diamond.results.json`.
"""

from __future__ import annotations

from pathlib import Path

from spar.eval.cache import CompletionCache
from spar.eval.orchestrator import evaluate_model
from spar.eval.profile import Profile, StagePlan, StageSampling
from spar.harness.model_grader import StubModelGrader
from spar.harness.user_sim import ScriptedUserSim, UserResponse

from tests.eval._fakes import FakeCompletion, abort_sample, make_model


def _run(profile: Profile, out: Path) -> Path:
    evaluate_model(
        make_model(id="naming"), profile,
        out_dir=out, cache=CompletionCache(out / "cache"), budget_usd=None, concurrency=1,
        responder=ScriptedUserSim(UserResponse(decision="deny")), grader=StubModelGrader(),
        completion_fn=FakeCompletion(mode="abort"),
        samples_for=lambda _s: [abort_sample()], sleep=lambda _s: None,
    )
    return out / "naming"


def test_repeated_split_writes_per_stage_files(tmp_path: Path) -> None:
    profile = Profile(
        competence=StageSampling(temperature=0.0, top_p=1.0, max_tokens=64, seed=7),
        reliability=StageSampling(temperature=0.7, top_p=1.0, max_tokens=64, seed=7),
        plan=[
            StagePlan(split="lite", k=1, stage="competence", published=False),
            StagePlan(split="lite", k=4, stage="reliability", published=False),
        ],
    )
    d = _run(profile, tmp_path / "run")
    assert (d / "lite.competence.results.json").exists()
    assert (d / "lite.reliability.results.json").exists()
    assert not (d / "lite.results.json").exists()      # no ambiguous overwrite-prone file


def test_unique_split_keeps_bare_filename(tmp_path: Path) -> None:
    profile = Profile(
        competence=StageSampling(temperature=0.0, top_p=1.0, max_tokens=64, seed=7),
        reliability=StageSampling(temperature=0.7, top_p=1.0, max_tokens=64, seed=7),
        plan=[StagePlan(split="lite", k=1, stage="competence", published=False)],
    )
    d = _run(profile, tmp_path / "run")
    assert (d / "lite.results.json").exists()          # consolidate-compatible bare name
