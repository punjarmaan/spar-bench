"""The `spar` CLI (module 40 §5). M1 implements `run`; `grade`/`report` land in M6/M8."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import typer

from spar.agents.base import Agent
from spar.dataset.loader import load_split
from spar.harness.graders import SampleScore, score
from spar.harness.report import build_results
from spar.harness.runner import run_episode

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
