"""EM4 — the SINGLE live smoke test (spec §9). ONE real cheap model end-to-end through
evaluate_model → consolidate → write_leaderboard.

RUN ONLY: OPENROUTER_API_KEY=… uv run pytest -q -m frontier tests/eval/test_frontier_smoke.py
EXCLUDED from the per-commit gate by `-m "not frontier"`. Double-gated: the `frontier`
marker AND a skipif on OPENROUTER_API_KEY (so even `-m frontier` no-ops without a key).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# A single cheap open model via OpenRouter — keeps the live call ~cents.
SMOKE_MODEL_ROUTE = "openrouter/meta-llama/llama-3.3-70b-instruct"

pytestmark = [
    pytest.mark.frontier,
    pytest.mark.skipif(
        not os.environ.get("OPENROUTER_API_KEY"),
        reason="live smoke needs OPENROUTER_API_KEY (and the `llm` extra)",
    ),
]


def test_one_cheap_model_produces_a_leaderboard_row(tmp_path: Path) -> None:
    from spar.eval.cache import CompletionCache
    from spar.eval.consolidate import consolidate, write_leaderboard
    from spar.eval.live import default_completion_fn
    from spar.eval.models import ModelConfig
    from spar.eval.profile import Profile, StagePlan, StageSampling
    from spar.harness.model_grader import StubModelGrader
    from spar.harness.user_sim import LiteLLMUserSim

    out_dir = tmp_path / "runs"
    cache = CompletionCache(tmp_path / "cache")
    live_fn = default_completion_fn()

    model = ModelConfig(
        id="llama-3.3-70b",
        route=SMOKE_MODEL_ROUTE,
        supports_response_format=False,
        version_pin=f"{SMOKE_MODEL_ROUTE}@smoke",
        **{"class": "open"},
    )
    # The leaderboard consolidates the COMPETENCE (main) + RELIABILITY (redline) splits — a
    # lite-only run writes no main.results.json and cannot consolidate (spec §6: lite is dev,
    # never published). Run the published shape at k=1 (cheapest publishable end-to-end path; the
    # bundled main/redline splits are tiny, so this stays ~cents).
    profile = Profile(
        competence=StageSampling(temperature=0.0, top_p=1.0, max_tokens=512, seed=7),
        reliability=StageSampling(temperature=0.7, top_p=1.0, max_tokens=512),
        plan=[
            StagePlan(split="main", k=1, stage="competence", published=True),
            StagePlan(split="redline", k=1, stage="reliability", published=True),
        ],
    )
    # Pinned infra: live responder (temp=0) sharing the agent route; offline stub grader so
    # a malformed/refused agent still grades and the row is well-formed without a judge call.
    responder = LiteLLMUserSim(SMOKE_MODEL_ROUTE, completion_fn=live_fn)
    grader = StubModelGrader()

    from spar.eval.orchestrator import evaluate_model

    evaluate_model(
        model,
        profile,
        out_dir=out_dir,
        cache=cache,
        budget_usd=5.0,
        concurrency=2,
        responder=responder,
        grader=grader,
    )

    # A per-model results.json must have been written.
    results = list(out_dir.glob("**/*.results.json"))
    assert results, f"no results.json written under {out_dir}"

    entries = consolidate(out_dir)
    assert len(entries) == 1, "exactly one model evaluated"
    entry = entries[0]

    # Well-formed verified-OR-partial row with real confirmed agent cost.
    assert entry.model == "llama-3.3-70b"
    assert entry.cls == "open"
    assert entry.status in {"verified", "partial"}
    assert 0.0 <= entry.trust_score <= 1.0
    assert entry.cost_usd > 0.0, "confirmed agent inference cost must be > 0 after a live run"

    # The full publish path runs without error and emits the three artifacts.
    write_leaderboard(entries, tmp_path)
    assert (tmp_path / "leaderboard.json").exists()
    assert (tmp_path / "leaderboard.csv").exists()
    assert (tmp_path / "LEADERBOARD.md").exists()
