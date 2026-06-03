"""`spar eval --offline` must NOT share a cache namespace with live runs.

The completion cache is content-addressed by (model, messages, sampling, trial_index) — it does
NOT distinguish an offline-stub response from a real paid one. So if an `--offline` pre-flight
and a live run share a `--cache-dir`, the live run silently cache-HITS the stub `abort` responses
at $0 and never calls the model (observed in Rung-1: a live run returned cost $0.00, all-refused,
instantly). The offline path therefore isolates its cache under a `_offline` subdir so a live run
with the same `--cache-dir` can never be satisfied by stub entries.
"""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from spar.harness.run_eval import app


def test_offline_eval_cache_is_isolated_from_live_namespace(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "eval",
            "--models", "configs/models.toml",
            "--profile", "configs/profile.lite.toml",
            "--only", "llama-4-maverick",
            "--cache-dir", str(cache),
            "--out-dir", str(tmp_path / "runs"),
            "--offline",
        ],
    )
    assert result.exit_code == 0, result.output

    # Offline stub completions were cached (the run did real work)...
    offline_entries = list((cache / "_offline").glob("*.json"))
    assert offline_entries, "offline run should have populated the isolated _offline cache"

    # ...but NOT in the top-level namespace a live run (`--cache-dir cache`) reads, so a live
    # run can never be silently satisfied by these stub responses.
    live_namespace_entries = list(cache.glob("*.json"))
    assert not live_namespace_entries, (
        "offline stub responses must not pollute the live cache namespace; found "
        f"{[p.name for p in live_namespace_entries]}"
    )
