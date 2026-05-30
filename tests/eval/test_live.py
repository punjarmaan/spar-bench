"""EM4 — offline tests for the live agent completion_fn wiring (no real model calls)."""

from __future__ import annotations


def test_default_completion_fn_returns_callable() -> None:
    # Importing spar.eval.live must NOT require litellm; only calling
    # default_completion_fn() resolves it. If litellm is unavailable the call
    # raises ImportError — but in the dev/llm environment it returns a callable.
    from spar.eval.live import default_completion_fn

    fn = default_completion_fn()
    assert callable(fn)


def test_live_module_imports_without_calling_litellm() -> None:
    # The module-level import is lazy: importing it must never touch litellm.
    import importlib

    mod = importlib.import_module("spar.eval.live")
    assert hasattr(mod, "default_completion_fn")
    # Nothing at module scope may have bound litellm.completion eagerly.
    assert "litellm" not in dir(mod)


def test_run_eval_injects_fake_completion_fn_and_pinned_infra(monkeypatch, tmp_path) -> None:
    import spar.harness.run_eval as re
    from spar.eval.models import ModelConfig
    from spar.eval.profile import DEFAULT_PROFILE
    from spar.harness.model_grader import StubModelGrader
    from spar.harness.user_sim import ScriptedUserSim, UserResponse

    captured: dict[str, object] = {}

    def fake_evaluate_model(model, profile, *, out_dir, cache, budget_usd,
                            concurrency, responder, grader, completion_fn, samples_for) -> None:
        captured.update(
            model=model, profile=profile, out_dir=out_dir, budget_usd=budget_usd,
            concurrency=concurrency, responder=responder, grader=grader,
            completion_fn=completion_fn,
        )

    monkeypatch.setattr(re, "evaluate_model", fake_evaluate_model)

    def fake_completion_fn(**kwargs):
        raise AssertionError("fake completion_fn must not be invoked by _run_eval wiring")

    model = ModelConfig(id="m", route="openrouter/test/model", **{"class": "open"})
    responder = ScriptedUserSim(UserResponse(decision="approve"))
    grader = StubModelGrader()

    re._run_eval(
        models=[model], profile=DEFAULT_PROFILE, out_dir=tmp_path / "runs",
        cache_dir=tmp_path / "cache", budget_usd=1.0, concurrency=2,
        agent_completion_fn=fake_completion_fn, responder=responder, grader=grader, only=None,
    )
    assert captured["model"] is model
    assert captured["responder"] is responder
    assert captured["grader"] is grader
    assert captured["completion_fn"] is fake_completion_fn   # the agent call layer IS threaded
    assert captured["budget_usd"] == 1.0
    assert captured["concurrency"] == 2
    assert captured["out_dir"] == tmp_path / "runs"


def test_run_eval_honours_only_filter(monkeypatch, tmp_path) -> None:
    import spar.harness.run_eval as re
    from spar.eval.models import ModelConfig
    from spar.eval.profile import DEFAULT_PROFILE
    from spar.harness.model_grader import StubModelGrader
    from spar.harness.user_sim import ScriptedUserSim, UserResponse

    seen: list[str] = []

    def fake_evaluate_model(model, profile, *, out_dir, cache, budget_usd,
                            concurrency, responder, grader, completion_fn, samples_for) -> None:
        seen.append(model.id)

    monkeypatch.setattr(re, "evaluate_model", fake_evaluate_model)
    a = ModelConfig(id="alpha", route="openrouter/test/a", **{"class": "open"})
    b = ModelConfig(id="beta", route="openrouter/test/b", **{"class": "frontier"})
    re._run_eval(
        models=[a, b], profile=DEFAULT_PROFILE, out_dir=tmp_path / "runs",
        cache_dir=tmp_path / "cache", budget_usd=None, concurrency=1,
        agent_completion_fn=lambda **kw: None,
        responder=ScriptedUserSim(UserResponse(decision="approve")),
        grader=StubModelGrader(), only="beta",
    )
    assert seen == ["beta"]


def test_run_eval_threads_samples_for(monkeypatch, tmp_path) -> None:
    import spar.harness.run_eval as re
    from spar.eval.models import ModelConfig
    from spar.eval.profile import DEFAULT_PROFILE
    from spar.harness.model_grader import StubModelGrader
    from spar.harness.user_sim import ScriptedUserSim, UserResponse

    captured: dict[str, object] = {}

    def fake_evaluate_model(model, profile, *, out_dir, cache, budget_usd, concurrency,
                            responder, grader, completion_fn, samples_for) -> None:
        captured["samples_for"] = samples_for

    monkeypatch.setattr(re, "evaluate_model", fake_evaluate_model)

    def sentinel(_split: str) -> list:
        return []

    re._run_eval(
        models=[ModelConfig(id="m", route="openrouter/test/m", **{"class": "open"})],
        profile=DEFAULT_PROFILE, out_dir=tmp_path / "runs", cache_dir=tmp_path / "cache",
        budget_usd=None, concurrency=1, agent_completion_fn=lambda **kw: None,
        responder=ScriptedUserSim(UserResponse(decision="deny")), grader=StubModelGrader(),
        only=None, samples_for=sentinel,
    )
    assert captured["samples_for"] is sentinel
