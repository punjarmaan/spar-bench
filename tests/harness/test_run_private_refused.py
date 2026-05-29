"""M8 Task 7b: the local CLI refuses `--split private` (C5 — never local over hidden gold)."""

from __future__ import annotations

from typer.testing import CliRunner

from spar.harness.run_eval import app


def test_run_refuses_private_split():
    result = CliRunner().invoke(app, [
        "run", "--split", "private",
        "--agent", "spar.agents.reference_agents:AlwaysAbortAgent", "--out", "/tmp/x.json",
    ])
    assert result.exit_code != 0
    assert "private" in result.output.lower()
    assert "server" in result.output.lower() or "not served" in result.output.lower()


def test_run_still_accepts_public_splits_help():
    out = CliRunner().invoke(app, ["run", "--help"]).output
    assert "lite" in out
