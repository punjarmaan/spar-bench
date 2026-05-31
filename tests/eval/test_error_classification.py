"""Task 4.2: context-overflow is a SCORED capability failure (audit S1).

Context-overflow is a deterministic capability failure — retrying just re-pays with no
different outcome.  Per S1 it must be treated as a capability/abort outcome that LANDS IN the
scored population (a failure that counts), NOT silently excluded from the denominator.  These
tests verify (a) classification (context-overflow is not infra), (b) that a context-overflow
propagating out of run_episode does not crash the eval run, and (c) that the sample is SCORED
as a capability failure (status MALFORMED_ACTION, sscore present, trials_c == 0,
catastrophic_class None) and appears in the scored population.
"""

from __future__ import annotations

from typing import Any

from spar.eval.cache import CompletionCache
from spar.eval.cost import CostMeter
from spar.eval.orchestrator import (
    SampleStatus,
    _is_infra_error,
    _run_sample,
    evaluate_model,
)
from spar.eval.profile import DEFAULT_PROFILE
from spar.harness.model_grader import StubModelGrader
from spar.harness.user_sim import ScriptedUserSim, UserResponse
from tests.eval._fakes import abort_sample, make_model


# ---------------------------------------------------------------------------
# Classification: context-overflow must NOT be an infra error
# ---------------------------------------------------------------------------


def test_context_overflow_is_not_infra_retryable() -> None:
    # A deterministic context-overflow re-fails identically; retrying just re-pays.
    assert _is_infra_error(Exception("context_length_exceeded: 200000 > 128000")) is False
    assert _is_infra_error(Exception("maximum context length is 128000")) is False


def test_rate_limit_and_timeout_stay_infra() -> None:
    assert _is_infra_error(Exception("429 rate limit")) is True
    assert _is_infra_error(TimeoutError("timed out")) is True


# ---------------------------------------------------------------------------
# Non-crash: a context-overflow from run_episode must not crash the eval run
# ---------------------------------------------------------------------------


class _ContextOverflowCompletion:
    """A completion_fn that always raises a context-length error (simulates a model
    whose prompt exceeds its context window)."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, *, model: str, messages: list[dict[str, Any]], **sampling: Any) -> Any:
        self.calls += 1
        raise RuntimeError("context_length_exceeded: prompt 200000 > 128000")


def _responder() -> ScriptedUserSim:
    return ScriptedUserSim(UserResponse(decision="deny"))


def test_context_overflow_run_sample_is_scored_capability_failure(tmp_path: Any) -> None:
    """A context-overflow in run_episode must not crash _run_sample and must be SCORED as a
    capability failure (audit S1): status MALFORMED_ACTION, a real SampleScore (NOT None) with
    trials_c == 0, and catastrophic_class None — i.e. it LANDS IN the scored population as a
    failure rather than being silently excluded as ERRORED_INFRA."""
    cache = CompletionCache(tmp_path)
    meter = CostMeter(budget_usd=None)
    completion = _ContextOverflowCompletion()

    status, sscore = _run_sample(
        abort_sample(), make_model(),
        sampling=DEFAULT_PROFILE.competence,
        completion_fn=completion,
        cache=cache, meter=meter, responder=_responder(), grader=StubModelGrader(),
        k=1, retries=2, sleep=lambda _s: None,
    )
    # (a) not retried: _is_infra_error is False for context-overflow, so the raw completion is
    # invoked exactly ONCE (no doomed re-tries), then the abort-scoring run makes NO model call.
    assert completion.calls == 1
    # (b) no crash + (c) SCORED as a capability failure into the scored population.
    assert status is SampleStatus.MALFORMED_ACTION
    assert sscore is not None
    assert sscore.trials_n == 1
    assert sscore.trials_c == 0           # solved 0 of k -> capability failure
    assert sscore.catastrophic_class is None  # no forbidden completion happened
    assert sscore.score == 0.0


def test_context_overflow_evaluate_model_completes_run(tmp_path: Any) -> None:
    """evaluate_model must complete (not raise) even when every sample overflows context."""
    out_dir = tmp_path / "runs"
    cache = CompletionCache(tmp_path / "cache")
    from spar.eval.profile import Profile, StagePlan, StageSampling

    profile = Profile(
        competence=StageSampling(temperature=0.0, top_p=1.0, max_tokens=64, seed=7),
        reliability=StageSampling(temperature=0.7, top_p=1.0, max_tokens=64, seed=7),
        plan=[StagePlan(split="lite", k=1, stage="competence", published=True)],
    )

    # Must not raise.
    evaluate_model(
        make_model(id="overflow-model"), profile,
        out_dir=out_dir, cache=cache, budget_usd=None, concurrency=1,
        responder=_responder(), grader=StubModelGrader(),
        completion_fn=_ContextOverflowCompletion(),
        retries=2, sleep=lambda _s: None,
    )
    import json
    manifest = json.loads(
        (out_dir / "overflow-model" / "run_manifest.json").read_text()
    )
    # The split must be present and the run must have been attempted.
    assert "lite" in manifest["splits"]
    split = manifest["splits"]["lite"]
    assert split["n"] >= 1
    # S1: an overflow lands IN the scored population (a counted capability failure), so it is
    # NOT excluded — n_scored matches n and scored_fraction is full (no silent denominator drop).
    assert split["n_scored"] == split["n"]
    assert split["scored_fraction"] == 1.0
    assert split["tally"].get("malformed_action", 0) == split["n"]
