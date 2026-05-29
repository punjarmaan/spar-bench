"""`spar-submit` — the leaderboard submission CLIENT (module 50 §3, C5).

Submits a TRAJECTORY-REPLAY `predictions.jsonl` to the Spar-Private leaderboard for
server-side grading. C5 (binding): the client offers NO `--agent module:Class` flag — the
server replays recorded trajectories only and NEVER imports attacker code in the process
holding the private split + hidden gold + canary. A single static trajectory per sample yields
pass^1 only (`pass_4` is null, F7); supplying k>=4 trajectories per sample enables pass^4.

The Private leaderboard SERVER (network/FS-isolated sandbox, real per-token quotas, redacted
per-sample results) is DESCOPED from v1 — only this client ships. Until the server exists,
`spar run --split private` is refused locally (C5). The HTTP boundary below is a thin,
stubbable seam so CI never makes a network call.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, NoReturn

import typer

app = typer.Typer(add_completion=False, help="Submit a trajectory-replay predictions.jsonl "
                                             "to the Spar-Private leaderboard (server-side grading).")

_DEFAULT_ENDPOINT = "https://leaderboard.spar.eval"


def classify_predictions(path: Path) -> dict[str, Any]:
    """Inspect a predictions.jsonl: max trajectories per sample + whether pass^4 is available.

    A single static trajectory per sample is pass^1 only (F7); >=4 enable the pass^4 estimate.
    """
    counts: dict[str, int] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        counts[rec["sample_id"]] = counts.get(rec["sample_id"], 0) + 1
    max_per_sample = max(counts.values()) if counts else 0
    return {
        "n_samples": len(counts),
        "max_trajectories_per_sample": max_per_sample,
        "pass_4_available": max_per_sample >= 4,
    }


def _server_unavailable() -> NoReturn:
    raise RuntimeError(
        "the Spar-Private leaderboard server is not yet available (descoped from v1). "
        "Trajectory-replay submission will open once the isolated grading server ships."
    )


def _post_submission(*, predictions: Path, model_name: str, endpoint: str, token: str) -> dict[str, Any]:
    """Thin HTTP seam: POST the trajectory-replay payload. Stubbed in tests; descoped in v1."""
    return _server_unavailable()


def _get_quota(*, endpoint: str, token: str) -> dict[str, Any]:
    """Thin HTTP seam: GET the caller's REAL remaining submission quota (no echo stub)."""
    return _server_unavailable()


def _get_status(*, run_id: str, endpoint: str, token: str) -> dict[str, Any]:
    """Thin HTTP seam: GET a run's REAL grading state (no echo stub)."""
    return _server_unavailable()


@app.callback(invoke_without_command=True)
def submit(
    ctx: typer.Context,
    predictions: Path = typer.Option(None, "--predictions",
                                     help="predictions.jsonl of recorded {sample_id, trajectory}"),
    model_name: str = typer.Option(None, "--model-name", help="model name for the leaderboard row"),
    endpoint: str = typer.Option(_DEFAULT_ENDPOINT, "--endpoint"),
    token: str = typer.Option("", "--token", help="leaderboard API token"),
) -> None:
    """Submit a trajectory-replay predictions.jsonl (the default action; no --agent path, C5).

    Curated frontier results are landed via a reviewed PR to the `spar-experiments` repo.
    """
    if ctx.invoked_subcommand is not None:
        return
    if predictions is None or model_name is None:
        typer.echo("error: --predictions and --model-name are required")
        raise typer.Exit(code=2)
    info = classify_predictions(predictions)
    if not info["pass_4_available"]:
        typer.echo(
            "note: a single static trajectory per sample yields pass_1 only; pass_4 is null "
            "(F7). Supply >=4 trajectories per sample (distinct trial_index) to enable pass_4."
        )
    result = _post_submission(predictions=predictions, model_name=model_name,
                              endpoint=endpoint, token=token)
    typer.echo(f"submitted {model_name}: run_id={result['run_id']}")


@app.command()
def quota(
    endpoint: str = typer.Option(_DEFAULT_ENDPOINT, "--endpoint"),
    token: str = typer.Option("", "--token"),
) -> None:
    """Show the caller's REAL remaining submission quota (server-enforced, not an echo stub)."""
    q = _get_quota(endpoint=endpoint, token=token)
    typer.echo(f"quota: {q['remaining']} of {q['limit']} submissions remaining")


@app.command()
def status(
    run_id: str = typer.Option(..., "--run-id"),
    endpoint: str = typer.Option(_DEFAULT_ENDPOINT, "--endpoint"),
    token: str = typer.Option("", "--token"),
) -> None:
    """Show a submitted run's REAL grading state (server-reported, not an echo stub)."""
    s = _get_status(run_id=run_id, endpoint=endpoint, token=token)
    typer.echo(f"run {s['run_id']}: {s['state']}")
