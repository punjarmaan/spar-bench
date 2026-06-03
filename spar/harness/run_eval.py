"""The `spar` CLI (module 40 §5). M1 implements `run`; `grade`/`report` land in M6/M8."""

from __future__ import annotations

import functools
import importlib
import json
import os
from collections.abc import Callable
from pathlib import Path

import typer

from typing import Any

from spar.agents.base import Agent
from spar.dataset.build_cli import build_cmd
from spar.dataset.loader import load_graded_split, load_split
from spar.eval.cache import CompletionCache
from spar.eval.cost import estimate_cost
from spar.eval.live import CompletionFn, default_completion_fn, load_env
from spar.eval.models import ModelConfig, load_models
from spar.eval.orchestrator import evaluate_model
from spar.eval.profile import Profile, load_profile
from spar.harness.graders import ModelGrader, SampleScore, score
from spar.harness.model_grader import LiteLLMModelGrader, StubModelGrader
from spar.harness.report import build_results, recompute_summary
from spar.harness.runner import run_episode
from spar.harness.user_sim import LiteLLMUserSim, ScriptedUserSim, UserResponse, UserSim
from spar.simulator.contract import Abort, Action, parse_action
from spar.simulator.schemas import Sample

app = typer.Typer(add_completion=False, help="Spar — payment-execution benchmark")

# `spar build` cuts a frozen dataset release (module 30 §5).
app.command(name="build")(build_cmd)


@app.callback()
def _callback() -> None:
    """Spar — payment-execution benchmark CLI."""


def _load_agent(spec: str) -> Agent:
    module_name, class_name = spec.split(":", 1)
    cls = getattr(importlib.import_module(module_name), class_name)
    agent: Agent = cls()
    return agent


def _make_grader(grader_model: str | None) -> ModelGrader:
    """H4: the live CLI ALWAYS supplies a Tier-C model grader so a gray-zone semantic sample
    grades instead of crashing with NotImplementedError.

    Default is the deterministic, offline `StubModelGrader` (pinned id recorded in
    results.json) so a local run is reproducible and needs no API. `--grader-model <id>` opts
    into the pinned LiteLLM judge (temp=0, F12) for a published-grade run.
    """
    if grader_model is None:
        return StubModelGrader()
    return LiteLLMModelGrader(grader_model)


def _unpriced_warnings(
    models: list[ModelConfig], *, only: str | None, budget_usd: float | None
) -> list[str]:
    """One warning per to-be-run model that is UNPRICED while a budget cap is set. An unpriced
    model meters at $0 whenever the provider omits response_cost, so the cap silently can't fire.
    Returns [] when no budget is set (the cap is moot) or every selected model is priced."""
    if budget_usd is None:
        return []
    out: list[str] = []
    for m in models:
        if only is not None and m.id != only:
            continue
        if m.price_in_per_mtok is None or m.price_out_per_mtok is None:
            out.append(
                f"WARNING: model '{m.id}' has no price_in/out_per_mtok in the roster. The "
                f"${budget_usd:g} budget cap then relies ENTIRELY on provider-reported cost; if "
                "the provider omits it, spend meters at $0 and the cap will NOT fire. Add prices "
                "to models.toml to guarantee the cap."
            )
    return out


def _run_eval(
    *,
    models: list[ModelConfig],
    profile: Profile,
    out_dir: Path,
    cache_dir: Path,
    budget_usd: float | None,
    concurrency: int,
    agent_completion_fn: CompletionFn,
    responder: UserSim,
    grader: ModelGrader,
    only: str | None,
    samples_for: Callable[[str], list[Sample]] | None = None,
) -> None:
    """Pure, offline-testable wiring: run evaluate_model for each selected model, threading the
    injected agent completion_fn + pinned responder/grader through the shared completion cache.
    The Typer `eval` command builds the live infra (default_completion_fn + LiteLLMUserSim) and
    delegates here; tests inject fakes. No litellm import in this function."""
    cache = CompletionCache(cache_dir)
    for model in models:
        if only is not None and model.id != only:
            continue
        evaluate_model(
            model, profile,
            out_dir=out_dir, cache=cache, budget_usd=budget_usd, concurrency=concurrency,
            responder=responder, grader=grader, completion_fn=agent_completion_fn,
            samples_for=samples_for,
        )


@app.command()
def run(
    split: str = typer.Option(..., help="lite | main | diamond | probe  (private is server-only, C5)"),
    agent: str = typer.Option(..., help="module:Class implementing the Agent protocol "
                                        "(TRUSTED LOCAL import only)"),
    out: Path = typer.Option(Path("results.json")),
    grader_model: str = typer.Option(
        None, help="Tier-C grader: omit for the offline StubModelGrader; pass a LiteLLM "
                   "model id for the pinned LLM judge (temp=0, F12)."),
) -> None:
    if split == "private":
        # C5: the local --agent path imports code in-process and must NEVER run against the
        # private split + hidden gold + canary. Private is served only by the trajectory-replay
        # leaderboard server (descoped from v1), not the local CLI.
        typer.echo(
            "refused: --split private is not served by the local CLI (C5). The local "
            "--agent import is trusted-local-only and must never run against the private "
            "split + hidden gold + canary. Use the trajectory-replay leaderboard server instead."
        )
        raise typer.Exit(code=2)
    samples = load_split(split)
    grader = _make_grader(grader_model)
    scores: list[SampleScore] = []
    canary = samples[0].canary if samples else "spar:none"
    for sample in samples:
        trace = run_episode(sample, _load_agent(agent), trial_index=0)
        scores.append(score(sample, trace, model_grader=grader))
    results = build_results(scores, split=split, canary=canary, build_seed=0,
                            weights={"score_floor": -1.0})
    out.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    typer.echo(f"wrote {out} — trust_score={results['summary']['trust_score']:.3f}")


class _ReplayAgent:
    """Replays a recorded trajectory action-by-action; aborts if it runs out of actions."""

    def __init__(self, trajectory: list[dict[str, Any]]) -> None:
        self._actions: list[Action] = [parse_action(a) for a in trajectory]
        self._i = 0

    def act(self, observation: object) -> Action:
        if self._i >= len(self._actions):
            return Abort(tool="abort", reason="trajectory_exhausted")
        action = self._actions[self._i]
        self._i += 1
        return action


@app.command()
def grade(
    predictions: Path = typer.Option(..., help="predictions.jsonl: {sample_id, trajectory}"),
    split: str = typer.Option(..., help="lite | main | diamond | probe | private"),
    out: Path = typer.Option(Path("results.json")),
    grader_model: str = typer.Option(
        None, help="Tier-C grader: omit for the offline StubModelGrader; pass a LiteLLM "
                   "model id for the pinned LLM judge (temp=0, F12)."),
) -> None:
    """Grade a pre-recorded predictions.jsonl by replaying it against the canonical seed.

    A static trajectory desyncs under re-seeding (F7), so this path reports pass^1 only and
    emits pass_4: null.
    """
    samples = {s.sample_id: s for s in load_split(split)}
    grader = _make_grader(grader_model)
    by_id: dict[str, list[list[dict[str, Any]]]] = {}
    for line in predictions.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        by_id.setdefault(rec["sample_id"], []).append(rec["trajectory"])

    scores: list[SampleScore] = []
    canary = next(iter(samples.values())).canary if samples else "spar:none"
    for sample_id, trajectories in by_id.items():
        sample = samples[sample_id]
        # Static replay: canonical trial_index 0, single trajectory -> pass^1 only.
        trace = run_episode(sample, _ReplayAgent(trajectories[0]), trial_index=0)
        scores.append(score(sample, trace, model_grader=grader))
    results = build_results(scores, split=split, canary=canary, build_seed=0,
                            weights={"score_floor": -1.0})
    out.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    typer.echo(
        f"wrote {out} — pass_1={results['summary']['pass_1']} "
        f"pass_4={results['summary']['pass_4']}"
    )


@app.command()
def report(
    results: Path = typer.Option(..., help="path to an existing results.json"),
) -> None:
    """Recompute and print the summary from an existing results.json (no model calls)."""
    data = json.loads(results.read_text(encoding="utf-8"))
    summary = recompute_summary(data)
    for key, value in summary.items():
        typer.echo(f"{key}: {value}")


def _offline_completion_fn() -> Callable[..., Any]:
    """A no-network completion_fn for `spar eval --offline`: always emits a valid abort action."""
    import json as _json

    class _Msg:
        def __init__(self, c: str) -> None:
            self.content = c

    class _Choice:
        def __init__(self, c: str) -> None:
            self.message = _Msg(c)

    class _Resp:
        def __init__(self, c: str) -> None:
            self.choices = [_Choice(c)]
            self.usage = type("U", (), {"prompt_tokens": 100, "completion_tokens": 20})()
            self._hidden_params = {"response_cost": 0.0}

    def fn(*, model: str, messages: list[dict[str, Any]], **sampling: Any) -> _Resp:
        return _Resp(_json.dumps({"tool": "abort", "args": {"reason": "offline"}}))

    return fn


@app.command(name="eval")
def eval_models(
    models: Path = typer.Option(..., help="models.toml roster (identity only)"),
    profile: Path = typer.Option(..., help="profile.toml (per-stage sampling + split×k plan)"),
    only: str = typer.Option(None, help="run/refresh a single model id (idempotent via cache)"),
    budget_usd: float = typer.Option(None, help="hard per-model confirmed-cost cap (USD)"),
    concurrency: int = typer.Option(1, help="bounded per-provider fan-out (recorded in manifest)"),
    cache_dir: Path = typer.Option(Path(".eval_cache"), help="completion cache dir (resume)"),
    out_dir: Path = typer.Option(Path("runs"), help="output root: runs/<model>/…"),
    grader_model: str = typer.Option(
        None, help="Tier-C grader: omit for offline StubModelGrader; LiteLLM id for the judge"),
    responder_model: str = typer.Option(
        None, help="escalation responder: omit for offline ScriptedUserSim(deny); LiteLLM id else"),
    offline: bool = typer.Option(
        False, "--offline", help="use a no-network completion_fn (CI/dev; no live model calls)"),
    dataset_dir: Path = typer.Option(
        None, help="a `spar build --private-out` dir; resolve REAL graded splits (else toy fallback)"),
) -> None:
    """Run one or all models for a profile (design §5.5/§7). Writes per-model results +
    trajectories + manifest. Resumable via the completion cache."""
    # Auto-load a repo-root .env so OPENROUTER_API_KEY placed there reaches LiteLLM (which reads
    # it from os.environ). A shell-exported key still wins (override=False). Fail fast with a
    # clear message BEFORE any work if a live run has no key, rather than a mid-run auth error.
    env_path = load_env()
    if not offline and not os.environ.get("OPENROUTER_API_KEY"):
        typer.echo(
            "error: OPENROUTER_API_KEY is not set. Put it in a .env at the repo root "
            "(auto-loaded) or export it, then re-run. Use --offline for a no-network run.",
            err=True,
        )
        raise typer.Exit(code=1)
    if env_path and not offline:
        typer.echo(f"loaded environment from {env_path}")
    # Isolate the offline (stub) cache from the live namespace. The completion cache keys only on
    # (model, messages, sampling, trial_index) — it cannot tell a stub `abort` response from a real
    # paid one — so sharing a --cache-dir between `--offline` and a live run lets the live run
    # silently cache-HIT the stubs at $0 and never call the model. Namespacing offline under
    # `_offline` makes that impossible (a live run with the same --cache-dir never reads it).
    if offline:
        cache_dir = cache_dir / "_offline"
    roster = load_models(models)
    # Guard (audit S?/B-cost): an unpriced model meters at $0 if the provider omits response_cost,
    # silently defeating the budget cap. Warn loudly before any spend.
    for warning in _unpriced_warnings(roster, only=only, budget_usd=budget_usd):
        typer.echo(warning, err=True)
    prof = load_profile(profile)
    grader = _make_grader(grader_model)
    responder: UserSim = (
        LiteLLMUserSim(responder_model)
        if responder_model is not None
        else ScriptedUserSim(UserResponse(decision="deny"))
    )
    agent_completion_fn = _offline_completion_fn() if offline else default_completion_fn()
    samples_for: Callable[[str], list[Sample]] | None = (
        functools.partial(load_graded_split, base_dir=dataset_dir)
        if dataset_dir is not None else None
    )
    _run_eval(
        models=roster, profile=prof, out_dir=out_dir, cache_dir=cache_dir,
        budget_usd=budget_usd, concurrency=concurrency,
        agent_completion_fn=agent_completion_fn, responder=responder, grader=grader, only=only,
        samples_for=samples_for,
    )
    typer.echo(f"wrote runs under {out_dir}/")


@app.command(name="eval-cost")
def eval_cost(
    models: Path = typer.Option(..., help="models.toml roster"),
    profile: Path = typer.Option(..., help="profile.toml"),
    avg_turns: int = typer.Option(6, help="heuristic turns-per-episode for the estimate"),
    dataset_dir: Path = typer.Option(
        None, help="a `spar build --private-out` dir; size the estimate against REAL splits"),
) -> None:
    """Dry-run pre-flight cost ESTIMATE (no model calls; design §5.6). Sets the launch decision."""
    roster = load_models(models)
    prof = load_profile(profile)
    samples_for: Callable[[str], list[Sample]] | None = (
        functools.partial(load_graded_split, base_dir=dataset_dir)
        if dataset_dir is not None else None
    )
    est = estimate_cost(roster, prof, avg_turns=avg_turns, samples_for=samples_for)
    for model_id, usd in est.items():
        typer.echo(f"{model_id}\t${usd:.4f}")


@app.command()
def leaderboard(
    runs: Path = typer.Option(Path("runs"), help="directory of runs/<model>/ results (EM2 output)"),
    out_dir: Path = typer.Option(Path("."), help="where to write leaderboard.{json,csv,md} + manifest"),
) -> None:
    """Consolidate every runs/<model>/ result into the published leaderboard artifacts (EM3)."""
    from spar.eval.consolidate import consolidate, write_leaderboard

    entries = consolidate(runs)
    write_leaderboard(entries, out_dir)
    typer.echo(
        f"wrote {out_dir}/leaderboard.json + .csv + LEADERBOARD.md + manifest "
        f"({len(entries)} models)"
    )


@app.command()
def bundle(
    runs: Path = typer.Option(Path("runs"), help="run outputs root: <runs>/<model>/…"),
    out_dir: Path = typer.Option(Path("bundle"), help="static viewer bundle output dir"),
) -> None:
    """Build the static, lazy-loadable viewer bundle (index + per-episode files + schema + types)."""
    from spar.eval.bundle import build_bundle
    build_bundle(runs_dir=runs, out_dir=out_dir)
    typer.echo(f"wrote {out_dir}/index.json + per-model episodes + schema/ + types/")
