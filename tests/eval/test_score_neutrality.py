"""Trajectory enrichment is score-neutral (enrichment T6).

The capture enrichment (per-turn records, all-k trajectories, cached reasoning) is purely additive
to the scoring path. This pins the deterministic offline scoring outputs as golden literals so any
future change that silently perturbs a score or status count is caught. The offline path is fully
deterministic (FakeCompletion abort + StubModelGrader + ScriptedUserSim(deny) + fixed seed), so the
values below are stable run-to-run (verified). Trap/catastrophic behavior is covered separately by
tests/eval/test_orchestrator_smoke.py and the full suite."""

from __future__ import annotations

import json
from pathlib import Path

from spar.eval.cache import CompletionCache
from spar.eval.orchestrator import evaluate_model
from spar.harness.model_grader import StubModelGrader
from tests.eval._fakes import FakeCompletion, make_model
from tests.eval.test_orchestrator import _responder, _toy_profile


def test_offline_scores_unchanged_after_enrichment(tmp_path: Path) -> None:
    evaluate_model(
        make_model(id="fm"), _toy_profile(),
        out_dir=tmp_path / "runs", cache=CompletionCache(tmp_path / "cache"),
        budget_usd=None, concurrency=1,
        responder=_responder(), grader=StubModelGrader(),
        completion_fn=FakeCompletion(mode="abort", response_cost=0.001),
        retries=2, sleep=lambda _s: None,
    )

    def summary_of(name: str) -> dict:
        r = json.loads((tmp_path / "runs" / "fm" / name).read_text())
        s = r["summary"]
        return {
            "status_tally": r["status_tally"],
            "n_samples": s["n_samples"],
            "competence_mean": s["competence_mean"],
            "pass_1": s["pass_1"],
            "pass_4": s["pass_4"],
            "any_catastrophic_rate": s["any_catastrophic_rate"],
        }

    # Golden values captured from the deterministic offline run; must not drift.
    assert summary_of("lite.competence.results.json") == {
        "status_tally": {"scored": 1},
        "n_samples": 1,
        "competence_mean": 1.0,
        "pass_1": 1.0,
        "pass_4": None,
        "any_catastrophic_rate": None,
    }
    assert summary_of("lite.reliability.results.json") == {
        "status_tally": {"scored": 1},
        "n_samples": 1,
        "competence_mean": 1.0,
        "pass_1": 1.0,
        "pass_4": 1.0,
        "any_catastrophic_rate": None,
    }
