"""EM2 Task 12: `spar eval` + `spar eval-cost` on the existing Typer app (offline)."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from spar.harness.run_eval import app

runner = CliRunner()


def _models_toml(tmp_path) -> str:
    p = tmp_path / "models.toml"
    p.write_text(
        '[[model]]\n'
        'id = "fakecli"\n'
        'route = "fake/route"\n'
        'class = "open"\n'
        'price_in_per_mtok = 1.0\n'
        'price_out_per_mtok = 2.0\n'
        'version_pin = "fake/route@2026-05"\n',
        encoding="utf-8",
    )
    return str(p)


def _profile_toml(tmp_path) -> str:
    p = tmp_path / "profile.toml"
    p.write_text(
        "[competence]\ntemperature = 0.0\ntop_p = 1.0\nmax_tokens = 64\nseed = 7\n\n"
        "[reliability]\ntemperature = 0.7\ntop_p = 1.0\nmax_tokens = 64\nseed = 7\n\n"
        '[[plan]]\nsplit = "lite"\nk = 1\nstage = "competence"\npublished = true\n',
        encoding="utf-8",
    )
    return str(p)


def test_eval_cost_dry_run_prints_estimate_no_calls(tmp_path) -> None:
    result = runner.invoke(
        app,
        ["eval-cost", "--models", _models_toml(tmp_path), "--profile", _profile_toml(tmp_path)],
    )
    assert result.exit_code == 0, result.output
    assert "fakecli" in result.output


def test_eval_offline_writes_run(tmp_path) -> None:
    out_dir = tmp_path / "runs"
    result = runner.invoke(
        app,
        [
            "eval",
            "--models", _models_toml(tmp_path),
            "--profile", _profile_toml(tmp_path),
            "--out-dir", str(out_dir),
            "--cache-dir", str(tmp_path / "cache"),
            "--offline",
        ],
    )
    assert result.exit_code == 0, result.output
    manifest = json.loads((out_dir / "fakecli" / "run_manifest.json").read_text())
    assert manifest["model"] == "fakecli"
    assert (out_dir / "fakecli" / "lite.results.json").exists()
