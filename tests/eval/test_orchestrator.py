"""EM2 Tasks 5-11: orchestrator status taxonomy, resume, cost cap, manifest."""

from __future__ import annotations

from spar.eval.orchestrator import PUBLISHABILITY_FLOOR, SampleStatus


def test_constants_and_enum() -> None:
    assert PUBLISHABILITY_FLOOR == 0.98
    assert {s.value for s in SampleStatus} == {
        "scored", "errored_infra", "malformed_action", "refused"
    }
    assert SampleStatus.SCORED.value == "scored"


import pytest

from spar.eval.orchestrator import _is_infra_error, _with_retries


def test_infra_error_classification() -> None:
    assert _is_infra_error(RuntimeError("429 Too Many Requests"))
    assert _is_infra_error(RuntimeError("503 Service Unavailable"))
    assert _is_infra_error(TimeoutError("read timed out"))
    assert _is_infra_error(RuntimeError("context_length_exceeded: too many tokens"))
    assert not _is_infra_error(ValueError("plain bug"))


def test_with_retries_succeeds_after_transient_failures() -> None:
    slept: list[float] = []
    n = {"i": 0}

    def call() -> str:
        n["i"] += 1
        if n["i"] < 3:
            raise RuntimeError("503 Service Unavailable")
        return "ok"

    out = _with_retries(call, retries=5, sleep=slept.append, jitter=lambda: 0.0)
    assert out == "ok"
    assert n["i"] == 3
    assert len(slept) == 2          # two backoffs before the third (successful) attempt
    assert slept == sorted(slept)   # exponential -> non-decreasing


def test_with_retries_exhausts_and_reraises_infra() -> None:
    def call() -> str:
        raise RuntimeError("429 rate limited")

    with pytest.raises(RuntimeError):
        _with_retries(call, retries=2, sleep=lambda _s: None, jitter=lambda: 0.0)


def test_with_retries_propagates_non_infra_immediately() -> None:
    calls = {"i": 0}

    def call() -> str:
        calls["i"] += 1
        raise ValueError("real bug")

    with pytest.raises(ValueError):
        _with_retries(call, retries=5, sleep=lambda _s: None, jitter=lambda: 0.0)
    assert calls["i"] == 1   # not retried


from spar.eval.cache import CompletionCache
from spar.eval.cost import CostMeter
from spar.eval.orchestrator import _run_sample
from spar.harness.model_grader import StubModelGrader
from spar.harness.user_sim import ScriptedUserSim, UserResponse
from spar.eval.profile import DEFAULT_PROFILE
from tests.eval._fakes import FakeCompletion, abort_sample, make_model


def _responder() -> ScriptedUserSim:
    return ScriptedUserSim(UserResponse(decision="deny"))


def test_run_sample_scored_on_valid_abort(tmp_path) -> None:
    cache = CompletionCache(tmp_path)
    meter = CostMeter(budget_usd=None)
    status, sscore = _run_sample(
        abort_sample(), make_model(),
        sampling=DEFAULT_PROFILE.competence,
        completion_fn=FakeCompletion(mode="abort", response_cost=0.03),
        cache=cache, meter=meter, responder=_responder(), grader=StubModelGrader(),
        k=1, retries=2, sleep=lambda _s: None,
    )
    assert status is SampleStatus.SCORED
    assert sscore is not None
    assert sscore.trials_n == 1
    assert meter.spent() > 0.0


def test_run_sample_errored_infra_after_retries_has_no_score(tmp_path) -> None:
    cache = CompletionCache(tmp_path)
    meter = CostMeter(budget_usd=None)
    raw = FakeCompletion(mode="raise_429")
    status, sscore = _run_sample(
        abort_sample(), make_model(),
        sampling=DEFAULT_PROFILE.competence,
        completion_fn=raw,
        cache=cache, meter=meter, responder=_responder(), grader=StubModelGrader(),
        k=1, retries=2, sleep=lambda _s: None,
    )
    assert status is SampleStatus.ERRORED_INFRA
    assert sscore is None
    assert raw.raised == 3            # initial attempt + 2 retries
