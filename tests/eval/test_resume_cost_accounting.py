"""Resume cost accounting: a cached re-run bills ~$0 but reports the same gross (uncached) cost.

Before the dual-accounting fix, a cache HIT re-recorded its stored response_cost into the meter, so
a resumed run over-reported cost_usd and could falsely trip the budget cap. Now the manifest carries
two figures: cost_usd = money ACTUALLY spent this run (cache misses), gross_cost_usd = what the run
would cost uncached (hits + misses). A fresh run has cost_usd == gross_cost_usd; a full resume has
cost_usd ~ 0 < gross_cost_usd.
"""

from __future__ import annotations

import json
from pathlib import Path

from spar.eval.cache import CompletionCache
from spar.eval.orchestrator import evaluate_model
from spar.eval.profile import Profile, StagePlan, StageSampling
from spar.harness.model_grader import StubModelGrader
from spar.harness.user_sim import ScriptedUserSim, UserResponse

from tests.eval._fakes import FakeCompletion, abort_sample, make_model


def _profile() -> Profile:
    return Profile(
        competence=StageSampling(temperature=0.0, top_p=1.0, max_tokens=64, seed=7),
        reliability=StageSampling(temperature=0.7, top_p=1.0, max_tokens=64, seed=7),
        plan=[StagePlan(split="lite", k=1, stage="competence", published=False)],
    )


def _run(out: Path, cache: CompletionCache) -> dict:
    evaluate_model(
        make_model(id="resume"), _profile(),
        out_dir=out, cache=cache, budget_usd=None, concurrency=1,
        responder=ScriptedUserSim(UserResponse(decision="deny")),
        grader=StubModelGrader(),
        completion_fn=FakeCompletion(mode="abort", response_cost=0.01),
        samples_for=lambda _split: [abort_sample()],
        sleep=lambda _s: None,
    )
    return json.loads((out / "resume" / "run_manifest.json").read_text())


def test_fresh_run_bills_equals_gross_then_resume_bills_zero(tmp_path: Path) -> None:
    cache = CompletionCache(tmp_path / "cache")

    first = _run(tmp_path / "run1", cache)
    # Fresh run (empty cache): every completion is a paid miss, so billed == gross > 0.
    assert first["cost_usd"] > 0
    assert first["cost_usd"] == first["gross_cost_usd"]

    second = _run(tmp_path / "run2", cache)  # SAME cache → every completion is a hit
    # Resume: nothing actually spent, but the uncached would-be cost is unchanged.
    assert second["cost_usd"] == 0.0
    assert second["gross_cost_usd"] == first["gross_cost_usd"]
    # Overhead is reported (0.0 here — offline stub responder/grader make no paid calls).
    assert first["overhead_cost_usd"] == 0.0


def test_overhead_helper_reads_live_responder_grader_cost() -> None:
    """_overhead_usd sums a live responder's + grader's cumulative cost_usd, and treats offline
    stubs (no cost_usd attribute) as $0."""
    from spar.eval.orchestrator import _overhead_usd

    class _Priced:
        cost_usd = 0.12

    class _Stub:
        pass

    assert _overhead_usd(_Priced(), _Priced()) == 0.24      # type: ignore[arg-type]
    assert _overhead_usd(_Stub(), _Stub()) == 0.0           # type: ignore[arg-type]
