"""The `spar` CLI (module 40 §5). M1 implements `run`; `grade`/`report` land in M6/M8."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import typer

from typing import Any

from spar.agents.base import Agent
from spar.dataset.loader import load_split
from spar.harness.graders import SampleScore, score
from spar.harness.report import build_results
from spar.harness.runner import run_episode
from spar.simulator.contract import Abort, Action, parse_action

app = typer.Typer(add_completion=False, help="Spar — payment-execution benchmark")


@app.callback()
def _callback() -> None:
    """Spar — payment-execution benchmark CLI."""


def _load_agent(spec: str) -> Agent:
    module_name, class_name = spec.split(":", 1)
    cls = getattr(importlib.import_module(module_name), class_name)
    agent: Agent = cls()
    return agent


@app.command()
def run(
    split: str = typer.Option(..., help="lite | main | diamond | private"),
    agent: str = typer.Option(..., help="module:Class implementing the Agent protocol"),
    out: Path = typer.Option(Path("results.json")),
) -> None:
    samples = load_split(split)
    scores: list[SampleScore] = []
    canary = samples[0].canary if samples else "spar:none"
    for sample in samples:
        trace = run_episode(sample, _load_agent(agent), trial_index=0)
        scores.append(score(sample, trace))
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
    split: str = typer.Option(..., help="lite | main | diamond | private"),
    out: Path = typer.Option(Path("results.json")),
) -> None:
    """Grade a pre-recorded predictions.jsonl by replaying it against the canonical seed.

    A static trajectory desyncs under re-seeding (F7), so this path reports pass^1 only and
    emits pass_4: null.
    """
    samples = {s.sample_id: s for s in load_split(split)}
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
        scores.append(score(sample, trace))
    results = build_results(scores, split=split, canary=canary, build_seed=0,
                            weights={"score_floor": -1.0})
    out.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    typer.echo(
        f"wrote {out} — pass_1={results['summary']['pass_1']} "
        f"pass_4={results['summary']['pass_4']}"
    )
