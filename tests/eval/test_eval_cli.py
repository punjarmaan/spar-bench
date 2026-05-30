"""EM2 Task 12: `spar eval` + `spar eval-cost` on the existing Typer app (offline)."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from spar.dataset.loader import load_split
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


def _tagged_private_release(tmp_path, n_main: int) -> Path:
    """Hand-write a tiny tagged private.jsonl with `n_main` real, gradeable 'main' samples."""
    priv = tmp_path / "priv"
    priv.mkdir(parents=True, exist_ok=True)
    base = load_split("lite")[0]
    samples = [base.model_copy(update={"split": "main", "sample_id": f"m{i}"}) for i in range(n_main)]
    (priv / "private.jsonl").write_text(
        "\n".join(s.model_dump_json() for s in samples) + "\n", encoding="utf-8"
    )
    return priv


def _main_profile_toml(tmp_path) -> str:
    p = tmp_path / "profile_main.toml"
    p.write_text(
        "[competence]\ntemperature = 0.0\ntop_p = 1.0\nmax_tokens = 64\nseed = 7\n\n"
        "[reliability]\ntemperature = 0.7\ntop_p = 1.0\nmax_tokens = 64\n\n"
        '[[plan]]\nsplit = "main"\nk = 1\nstage = "competence"\npublished = true\n',
        encoding="utf-8",
    )
    return str(p)


def test_eval_dataset_dir_runs_real_samples_not_toy(tmp_path) -> None:
    priv = _tagged_private_release(tmp_path, n_main=2)   # 2 real main samples (toy fallback = 1)
    out_dir = tmp_path / "runs"
    result = runner.invoke(
        app,
        [
            "eval",
            "--models", _models_toml(tmp_path),
            "--profile", _main_profile_toml(tmp_path),
            "--out-dir", str(out_dir),
            "--cache-dir", str(tmp_path / "cache"),
            "--dataset-dir", str(priv),
            "--offline",
        ],
    )
    assert result.exit_code == 0, result.output
    res = json.loads((out_dir / "fakecli" / "main.results.json").read_text())
    assert res["summary"]["n_samples"] == 2     # ran the 2 tagged 'main' samples, NOT the 1 toy


def test_eval_dataset_dir_bad_path_fails_loudly(tmp_path) -> None:
    # A bad --dataset-dir surfaces the resolver error instead of silently running the toy split.
    bad = runner.invoke(
        app,
        [
            "eval",
            "--models", _models_toml(tmp_path),
            "--profile", _main_profile_toml(tmp_path),
            "--out-dir", str(tmp_path / "runs"),
            "--cache-dir", str(tmp_path / "cache"),
            "--dataset-dir", str(tmp_path / "nonexistent"),
            "--offline",
        ],
    )
    assert bad.exit_code != 0


def test_eval_cost_dataset_dir_sizes_real_split(tmp_path) -> None:
    priv = _tagged_private_release(tmp_path, n_main=50)   # 50 real main samples
    result = runner.invoke(
        app,
        [
            "eval-cost",
            "--models", _models_toml(tmp_path),
            "--profile", _main_profile_toml(tmp_path),
            "--dataset-dir", str(priv),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "fakecli" in result.output

    # A bad --dataset-dir fails LOUDLY (no silent toy fallback for a pre-flight estimate).
    bad = runner.invoke(
        app,
        [
            "eval-cost",
            "--models", _models_toml(tmp_path),
            "--profile", _main_profile_toml(tmp_path),
            "--dataset-dir", str(tmp_path / "nonexistent"),
        ],
    )
    assert bad.exit_code != 0
