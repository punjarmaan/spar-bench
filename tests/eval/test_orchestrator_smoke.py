"""Offline orchestrator smoke test: full pipeline + catastrophic gate fires at $0.

Drives an always-complete trajectory through the REAL `evaluate_model` over a handful of
catastrophic-applicable CONSENT_MANDATE traps plus one non-catastrophic routing sample.
Proves the full pipeline (results.json, run_manifest.json, trajectories/*.jsonl) runs
end-to-end, the catastrophic gate fires (any_catastrophic_rate > 0), and pass^k is populated.
No live API calls.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from spar.dataset.applicability import applicability_from_world_config
from spar.dataset.generator import GenSpec, generate
from spar.eval.cache import CompletionCache
from spar.eval.orchestrator import evaluate_model
from spar.eval.profile import Profile, StagePlan, StageSampling
from spar.harness.model_grader import StubModelGrader
from spar.harness.user_sim import ScriptedUserSim, UserResponse
from spar.simulator.enums import Axis, Difficulty
from spar.simulator.schemas import Sample

from tests.eval._fakes import abort_sample, make_model

# ---------------------------------------------------------------------------
# Always-complete scripted sequence: select_route → submit_authorization →
# capture → abort (clamped).  Trips the catastrophic gate on consent_mandate
# traps that contain issuer_behavior.
# ---------------------------------------------------------------------------
_ALWAYS_COMPLETE_PATH: tuple[dict[str, Any], ...] = (
    {"tool": "select_route", "args": {"acquirer_id": "acq_a", "method": "visa"}},
    {"tool": "submit_authorization", "args": {"idempotency_key": "k1"}},
    {"tool": "capture", "args": {"idempotency_key": "k2"}},
    {"tool": "abort", "args": {"reason": "done"}},
)


class _Msg:
    def __init__(self, content: str) -> None:
        self.content = content


class _Choice:
    def __init__(self, content: str) -> None:
        self.message = _Msg(content)


class _Usage:
    def __init__(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class _Resp:
    def __init__(self, content: str) -> None:
        self.choices = [_Choice(content)]
        self.usage = _Usage(prompt_tokens=200, completion_tokens=4)
        self._hidden_params: dict[str, object] = {"response_cost": 0.0}


class _AlwaysCompleteCompletion:
    """Offline $0 completion_fn that drives select_route → submit_authorization → capture → abort.

    Stateless across agents: the step is inferred from the assistant-turn count in `messages`.
    Clamps at the last entry so no index error when the episode has more turns than the script.
    """

    def __call__(
        self, *, model: str, messages: list[dict[str, Any]], **sampling: Any
    ) -> _Resp:
        step = sum(1 for m in messages if m.get("role") == "assistant")
        action = _ALWAYS_COMPLETE_PATH[min(step, len(_ALWAYS_COMPLETE_PATH) - 1)]
        return _Resp(json.dumps(action))


# ---------------------------------------------------------------------------
# Smoke sample set
# ---------------------------------------------------------------------------

def _smoke_samples() -> list[Sample]:
    """3-4 stamped CONSENT_MANDATE/EASY traps + the abort_sample (non-catastrophic routing)."""
    traps: list[Sample] = []
    for seed in range(1, 5):
        sample = generate(
            GenSpec(axis=Axis.CONSENT_MANDATE, seed=seed, difficulty=Difficulty.EASY, is_trap=True)
        )
        # Stamp build-time applicability so the catastrophic gate is exercised.
        # generate() leaves gold.expected_violations empty; mirrors build._stamp_applicability.
        sample = sample.model_copy(
            update={
                "gold": sample.gold.model_copy(
                    update={
                        "expected_violations": sorted(
                            applicability_from_world_config(sample), key=lambda v: v.value
                        )
                    }
                )
            }
        )
        traps.append(sample)

    # Non-catastrophic routing sample: contributes to pass^k population (non-catastrophic)
    # and is solved by the always-complete fake (it reaches SETTLED/CLOSED), so pass_1 > 0.
    non_cat = abort_sample()
    return traps + [non_cat]


# ---------------------------------------------------------------------------
# The test
# ---------------------------------------------------------------------------

def test_orchestrator_smoke_pipeline_and_catastrophic_gate(tmp_path: Path) -> None:
    """Full pipeline runs end-to-end; catastrophic gate fires and summary keys are present."""
    samples = _smoke_samples()
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    model = make_model(id="smoke")
    profile = Profile(
        competence=StageSampling(temperature=0.0, top_p=1.0, max_tokens=2048, seed=7),
        reliability=StageSampling(temperature=0.7, top_p=1.0, max_tokens=2048, seed=7),
        plan=[
            StagePlan(split="lite", k=2, stage="reliability", published=False),
        ],
    )

    evaluate_model(
        model,
        profile,
        out_dir=out_dir,
        cache=CompletionCache(tmp_path / "cache"),
        budget_usd=None,
        concurrency=1,
        responder=ScriptedUserSim(UserResponse(decision="deny")),
        grader=StubModelGrader(),
        completion_fn=_AlwaysCompleteCompletion(),
        retries=0,
        sleep=lambda _: None,
        samples_for=lambda split: samples,
    )

    model_dir = out_dir / "smoke"

    # -- Assert 1: results.json exists and any_catastrophic_rate > 0 -------------------
    results_path = model_dir / "lite.results.json"
    assert results_path.exists(), f"results file not found at {results_path}"
    results: dict[str, Any] = json.loads(results_path.read_text(encoding="utf-8"))
    any_cat_rate = results["summary"]["any_catastrophic_rate"]
    assert isinstance(any_cat_rate, float), (
        f"any_catastrophic_rate should be a float, got {any_cat_rate!r}"
    )
    assert any_cat_rate > 0, (
        f"catastrophic gate did not fire: any_catastrophic_rate={any_cat_rate}"
    )

    # -- Assert 2: run_manifest.json exists --------------------------------------------
    manifest_path = model_dir / "run_manifest.json"
    assert manifest_path.exists(), f"run_manifest not found at {manifest_path}"

    # -- Assert 3: at least one trajectory JSONL written -------------------------------
    trajectories = list((model_dir / "trajectories").glob("*.jsonl"))
    assert len(trajectories) >= 1, (
        f"no trajectory files found under {model_dir / 'trajectories'}"
    )

    # -- Assert 4: classes_with_coverage is truthy (e.g. "3/7") -----------------------
    classes_coverage = results["summary"]["classes_with_coverage"]
    assert classes_coverage, (
        f"classes_with_coverage should be truthy, got {classes_coverage!r}"
    )
    # Confirm it looks like the expected "covered/total" format
    assert "/" in str(classes_coverage), (
        f"classes_with_coverage unexpected format: {classes_coverage!r}"
    )

    # -- Assert 5: pass_1 is not None --------------------------------------------------
    pass_1 = results["summary"]["pass_1"]
    assert pass_1 is not None, (
        "pass_1 is None — non-catastrophic sample did not land in pass^k population"
    )
