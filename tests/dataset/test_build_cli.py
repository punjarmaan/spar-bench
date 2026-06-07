"""`spar build` cuts public + private into SEPARATE directories."""

from __future__ import annotations

from typer.testing import CliRunner

from spar.harness.run_eval import app


def test_spar_build_writes_public_and_private_separately(tmp_path):
    pub = tmp_path / "public"
    priv = tmp_path / "private"
    result = CliRunner().invoke(app, [
        "build", "--seed", "12345", "--public-out", str(pub),
        "--private-out", str(priv), "--version", "1.0.0",
    ])
    assert result.exit_code == 0, result.output
    assert (pub / "main.jsonl").exists()
    assert (pub / "main.manifest.json").exists()
    assert (priv / "private.jsonl").exists()
    assert not (pub / "private.jsonl").exists()
