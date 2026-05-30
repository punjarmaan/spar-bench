"""EM2 Tasks 5-11: orchestrator status taxonomy, resume, cost cap, manifest."""

from __future__ import annotations

import json

import pytest

from spar.eval.cache import CompletionCache
from spar.eval.cost import CostMeter
from spar.eval.orchestrator import (
    PUBLISHABILITY_FLOOR,
    SampleStatus,
    _cached_completion_fn,
    _is_infra_error,
    _run_sample,
    _with_retries,
    evaluate_all,
    evaluate_model,
)
from spar.eval.profile import DEFAULT_PROFILE, Profile, StagePlan, StageSampling
from spar.harness.model_grader import StubModelGrader
from spar.harness.user_sim import ScriptedUserSim, UserResponse
from tests.eval._fakes import FakeCompletion, abort_sample, make_model


def test_constants_and_enum() -> None:
    assert PUBLISHABILITY_FLOOR == 0.98
    assert {s.value for s in SampleStatus} == {
        "scored", "errored_infra", "malformed_action", "refused"
    }
    assert SampleStatus.SCORED.value == "scored"


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


def test_cached_completion_fn_serves_second_call_from_cache(tmp_path) -> None:
    cache = CompletionCache(tmp_path)
    raw = FakeCompletion(mode="abort", response_cost=0.02)
    fn = _cached_completion_fn(raw, cache, retries=2, sleep=lambda _s: None)

    msgs = [{"role": "user", "content": "act"}]
    r1 = fn(model="fake/route", messages=msgs, temperature=0.0, top_p=1.0, max_tokens=64)
    assert raw.calls == 1
    content1 = r1.choices[0].message.content

    # Same request -> cache hit -> NO new underlying call, identical content + cost.
    r2 = fn(model="fake/route", messages=msgs, temperature=0.0, top_p=1.0, max_tokens=64)
    assert raw.calls == 1                       # not incremented
    assert r2.choices[0].message.content == content1
    assert r2._hidden_params["response_cost"] == 0.02


def test_pass_k_trials_not_collapsed_by_cache(tmp_path) -> None:
    """B1 regression: distinct trial indices must NOT share one cached completion (pass^k
    diversity preserved), yet re-running the same trial replays from cache (resume)."""
    cache = CompletionCache(tmp_path)
    raw = FakeCompletion(mode="counter", response_cost=0.0)
    msgs = [{"role": "user", "content": "act"}]
    fn0 = _cached_completion_fn(raw, cache, retries=2, sleep=lambda _s: None, trial_index=0)
    fn1 = _cached_completion_fn(raw, cache, retries=2, sleep=lambda _s: None, trial_index=1)
    r0 = fn0(model="m", messages=msgs, temperature=0.7)
    r1 = fn1(model="m", messages=msgs, temperature=0.7)
    assert raw.calls == 2                                   # two distinct cold calls, not collapsed
    assert r0.choices[0].message.content != r1.choices[0].message.content
    again0 = fn0(model="m", messages=msgs, temperature=0.7)
    assert raw.calls == 2                                   # trial 0 replays from cache, no new call
    assert again0.choices[0].message.content == r0.choices[0].message.content


def test_run_sample_malformed_is_scored_as_abort(tmp_path) -> None:
    cache = CompletionCache(tmp_path)
    meter = CostMeter(budget_usd=None)
    status, sscore = _run_sample(
        abort_sample(), make_model(),
        sampling=DEFAULT_PROFILE.competence,
        completion_fn=FakeCompletion(mode="garbage", response_cost=0.01),
        cache=cache, meter=meter, responder=_responder(), grader=StubModelGrader(),
        k=1, retries=2, sleep=lambda _s: None,
    )
    assert status is SampleStatus.MALFORMED_ACTION
    assert sscore is not None
    assert sscore.trials_n == 1


def _toy_profile() -> Profile:
    s = StageSampling(temperature=0.0, top_p=1.0, max_tokens=64, seed=7)
    r = StageSampling(temperature=0.7, top_p=1.0, max_tokens=64, seed=7)
    return Profile(
        competence=s, reliability=r,
        plan=[
            StagePlan(split="lite", k=1, stage="competence", published=True),
            StagePlan(split="lite", k=4, stage="reliability", published=True),
        ],
    )


def test_evaluate_model_writes_results_trajectories_and_manifest(tmp_path) -> None:
    model = make_model(id="fakemodel")
    out_dir = tmp_path / "runs"
    cache = CompletionCache(tmp_path / "cache")
    evaluate_model(
        model, _toy_profile(),
        out_dir=out_dir, cache=cache, budget_usd=None, concurrency=1,
        responder=_responder(), grader=StubModelGrader(),
        completion_fn=FakeCompletion(mode="abort", response_cost=0.001),
        retries=2, sleep=lambda _s: None,
    )
    base = out_dir / "fakemodel"
    res = json.loads((base / "lite.results.json").read_text())
    assert res["split"] == "lite"
    assert res["summary"]["n_samples"] >= 1
    traj_dir = base / "trajectories"
    traj_files = list(traj_dir.glob("*.jsonl"))
    assert traj_files
    first_line = traj_files[0].read_text().splitlines()[0]
    json.loads(first_line)
    manifest = json.loads((base / "run_manifest.json").read_text())
    for key in (
        "spar_version", "canary", "build_seed", "weights", "grader_model",
        "responder_model", "scaffold_version", "model_version_pin", "run_date",
        "cache_digest", "splits",
    ):
        assert key in manifest, key
    assert manifest["scaffold_version"]
    assert manifest["grader_model"] == "stub-model-grader@1"
    assert manifest["model_version_pin"] == "fake/route@2026-05"
    assert "lite" in manifest["splits"]
    assert manifest["splits"]["lite"]["status"] in ("verified", "partial")
    assert 0.0 <= manifest["splits"]["lite"]["scored_fraction"] <= 1.0


def test_evaluate_model_cap_marks_partial(tmp_path) -> None:
    model = make_model(id="capped")
    out_dir = tmp_path / "runs"
    cache = CompletionCache(tmp_path / "cache")
    evaluate_model(
        model, _toy_profile(),
        out_dir=out_dir, cache=cache, budget_usd=0.0005, concurrency=1,
        responder=_responder(), grader=StubModelGrader(),
        completion_fn=FakeCompletion(mode="abort", response_cost=0.01),
        retries=2, sleep=lambda _s: None,
    )
    manifest = json.loads((out_dir / "capped" / "run_manifest.json").read_text())
    assert manifest["budget_hit"] is True
    assert manifest["splits"]["lite"]["status"] == "partial"


def test_evaluate_model_resume_is_idempotent(tmp_path) -> None:
    model = make_model(id="resumed")
    out_dir = tmp_path / "runs"
    cache = CompletionCache(tmp_path / "cache")
    raw = FakeCompletion(mode="abort", response_cost=0.002)

    def run() -> dict:
        evaluate_model(
            model, _toy_profile(),
            out_dir=out_dir, cache=cache, budget_usd=None, concurrency=1,
            responder=_responder(), grader=StubModelGrader(),
            completion_fn=raw, retries=2, sleep=lambda _s: None,
        )
        return json.loads((out_dir / "resumed" / "lite.results.json").read_text())

    first = run()
    calls_after_first = raw.calls
    second = run()
    assert raw.calls == calls_after_first
    assert first["summary"]["trust_score"] == second["summary"]["trust_score"]
    assert first["per_sample"] == second["per_sample"]


def test_evaluate_model_empty_plan_writes_manifest_without_crash(tmp_path) -> None:
    model = make_model(id="emptyplan")
    out_dir = tmp_path / "runs"
    cache = CompletionCache(tmp_path / "cache")
    profile = Profile(
        competence=StageSampling(temperature=0.0, top_p=1.0, max_tokens=64, seed=7),
        reliability=StageSampling(temperature=0.7, top_p=1.0, max_tokens=64, seed=7),
        plan=[],
    )
    evaluate_model(
        model, profile,
        out_dir=out_dir, cache=cache, budget_usd=None, concurrency=1,
        responder=_responder(), grader=StubModelGrader(),
        completion_fn=FakeCompletion(mode="abort", response_cost=0.0),
        retries=2, sleep=lambda _s: None,
    )
    manifest = json.loads((out_dir / "emptyplan" / "run_manifest.json").read_text())
    assert manifest["splits"] == {}
    assert manifest["spar_version"]


def test_evaluate_all_runs_each_model_and_honors_only(tmp_path) -> None:
    a = make_model(id="alpha")
    b = make_model(id="beta")
    out_dir = tmp_path / "runs"
    cache = CompletionCache(tmp_path / "cache")
    evaluate_all(
        [a, b], _toy_profile(),
        out_dir=out_dir, cache=cache, budget_usd=None, concurrency=1,
        responder=_responder(), grader=StubModelGrader(),
        only="beta",
        completion_fn=FakeCompletion(mode="abort", response_cost=0.001),
        retries=2, sleep=lambda _s: None,
    )
    assert not (out_dir / "alpha").exists()
    assert (out_dir / "beta" / "run_manifest.json").exists()
    evaluate_all(
        [a, b], _toy_profile(),
        out_dir=out_dir, cache=cache, budget_usd=None, concurrency=1,
        responder=_responder(), grader=StubModelGrader(),
        completion_fn=FakeCompletion(mode="abort", response_cost=0.001),
        retries=2, sleep=lambda _s: None,
    )
    assert (out_dir / "alpha" / "run_manifest.json").exists()
    assert (out_dir / "beta" / "run_manifest.json").exists()


def test_exit_criterion_main_pass1_plus_diamond_pass4_offline(tmp_path) -> None:
    """EM2 exit: one model runs Main pass^1 + Diamond pass^4 offline, resumable, with confirmed
    cost + cap, writing results.json + trajectories + manifest."""
    model = make_model(id="exit")
    out_dir = tmp_path / "runs"
    cache = CompletionCache(tmp_path / "cache")
    s = StageSampling(temperature=0.0, top_p=1.0, max_tokens=64, seed=7)
    r = StageSampling(temperature=0.7, top_p=1.0, max_tokens=64, seed=7)
    profile = Profile(
        competence=s, reliability=r,
        plan=[
            StagePlan(split="lite", k=1, stage="competence", published=True),
            StagePlan(split="lite", k=4, stage="reliability", published=True),
        ],
    )
    raw = FakeCompletion(mode="abort", response_cost=0.001)
    evaluate_model(
        model, profile,
        out_dir=out_dir, cache=cache, budget_usd=10.0, concurrency=1,
        responder=_responder(), grader=StubModelGrader(),
        completion_fn=raw, retries=2, sleep=lambda _s: None,
    )
    base = out_dir / "exit"
    assert (base / "lite.results.json").exists()
    assert list((base / "trajectories").glob("*.jsonl"))
    manifest = json.loads((base / "run_manifest.json").read_text())
    assert manifest["cost_usd"] > 0.0
    assert manifest["budget_hit"] is False
    res = json.loads((base / "lite.results.json").read_text())
    assert res["k"] in (1, 4)
    calls = raw.calls
    evaluate_model(
        model, profile,
        out_dir=out_dir, cache=cache, budget_usd=10.0, concurrency=1,
        responder=_responder(), grader=StubModelGrader(),
        completion_fn=raw, retries=2, sleep=lambda _s: None,
    )
    assert raw.calls == calls
