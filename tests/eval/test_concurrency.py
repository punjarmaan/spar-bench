"""Tier-2b: concurrent sample workers must keep cost accounting EXACT (no lost updates).

The CostMeter and responder/grader cost are mutated by every episode; without locking, parallel
`_billed += cost` updates race and under-count spend, silently weakening the budget cap. These tests
fail (non-deterministically) if the locks are removed.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

from spar.eval.agent import CallUsage
from spar.eval.cache import CompletionCache
from spar.eval.cost import CostMeter
from spar.eval.models import ModelConfig
from spar.eval.orchestrator import evaluate_model
from spar.eval.profile import Profile, StagePlan, StageSampling
from spar.harness.model_grader import StubModelGrader
from spar.harness.user_sim import ScriptedUserSim, UserResponse

from tests.eval._fakes import FakeCompletion, abort_sample, make_model


def _model() -> ModelConfig:
    return ModelConfig(id="m", route="openrouter/x/y", cls="open")


def test_meter_record_is_thread_safe_no_lost_updates() -> None:
    """20 threads × 200 records of $0.01 = $40.00 EXACTLY. A lost-update race would under-count."""
    meter = CostMeter(budget_usd=None)
    m = _model()

    def worker() -> None:
        for _ in range(200):
            meter.record(CallUsage(prompt_tokens=0, completion_tokens=0, response_cost=0.01), m)

    threads = [threading.Thread(target=worker) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert abs(meter.spent() - 40.0) < 1e-9
    assert abs(meter.gross() - 40.0) < 1e-9


def _distinct_samples(n: int) -> list:
    # Distinct goals -> distinct system prompt -> distinct cache keys -> every call is a paid miss
    # (no cache collision), so the total billed is deterministic and exact.
    out = []
    for i in range(n):
        s = abort_sample()
        out.append(s.model_copy(update={
            "sample_id": f"spar_conc_{i}",
            "mandate": s.mandate.model_copy(update={"goal": f"buy distinct item {i}"}),
        }))
    return out


def test_evaluate_model_cost_is_exact_under_concurrency(tmp_path: Path) -> None:
    """End-to-end: 12 distinct samples through concurrency=4, each a 3-call happy path at $0.01.
    Total billed must be EXACTLY 12*3*$0.01 = $0.36 — proof the concurrent meter loses no cost."""
    samples = _distinct_samples(12)
    profile = Profile(
        competence=StageSampling(temperature=0.0, top_p=1.0, max_tokens=64, seed=7),
        reliability=StageSampling(temperature=0.7, top_p=1.0, max_tokens=64, seed=7),
        plan=[StagePlan(split="lite", k=1, stage="competence", published=False)],
    )
    evaluate_model(
        make_model(id="conc"), profile,
        out_dir=tmp_path, cache=CompletionCache(tmp_path / "cache"), budget_usd=None, concurrency=4,
        responder=ScriptedUserSim(UserResponse(decision="deny")), grader=StubModelGrader(),
        completion_fn=FakeCompletion(mode="abort", response_cost=0.01),
        samples_for=lambda _s: samples, sleep=lambda _s: None,
    )
    manifest = json.loads((tmp_path / "conc" / "run_manifest.json").read_text())
    assert manifest["splits"]["lite"]["n"] == 12
    assert manifest["splits"]["lite"]["n_scored"] == 12          # all processed under concurrency
    assert abs(manifest["cost_usd"] - 0.36) < 1e-9               # exact: no lost cost updates
    assert manifest["cost_usd"] == manifest["gross_cost_usd"]    # fresh run: billed == gross
    # one trajectory per distinct sample
    assert len(list((tmp_path / "conc" / "trajectories").glob("*.jsonl"))) == 12
