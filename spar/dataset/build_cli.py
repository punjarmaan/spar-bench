"""The `spar build` subcommand: cut a new versioned, canaried, frozen release.

Public projected splits and the server-side Private build go to SEPARATE directories (Private
isolation). Not used per eval run — eval loads the frozen artifacts.
"""

from __future__ import annotations

from pathlib import Path

import typer

from spar.dataset.build import build


def build_cmd(
    seed: int = typer.Option(..., "--seed", help="Build seed; fixes the splits (canary is fresh)."),
    public_out: Path = typer.Option(..., "--public-out", help="Dir for PUBLIC projected JSONL + manifests."),
    private_out: Path = typer.Option(..., "--private-out", help="SEPARATE dir for the server-side Private build."),
    version: str = typer.Option(..., "--version", help="spar_version recorded in manifests."),
) -> None:
    """Cut a frozen, canaried dataset release into separate public + private directories."""
    build(public_dir=public_out, private_dir=private_out, build_seed=seed,
          spar_version=version)
    typer.echo(
        f"built public splits into {public_out} and Private into {private_out} "
        f"(seed={seed}, version={version})"
    )
