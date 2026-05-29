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
