"""M8 Task 7: the spar-submit leaderboard CLIENT (trajectory-replay only, C5)."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from spar.leaderboard.submit import app, classify_predictions


def test_classify_single_trajectory_yields_passk_null(tmp_path):
    preds = tmp_path / "preds.jsonl"
    preds.write_text(
        json.dumps({"sample_id": "s1", "trajectory": [{"tool": "abort", "reason": "x"}]}) + "\n"
    )
    info = classify_predictions(preds)
    assert info["max_trajectories_per_sample"] == 1
    assert info["pass_4_available"] is False


def test_classify_multi_trajectory_enables_passk(tmp_path):
    preds = tmp_path / "preds.jsonl"
    lines = [
        json.dumps({"sample_id": "s1", "trial_index": i,
                    "trajectory": [{"tool": "abort", "reason": "x"}]})
        for i in range(4)
    ]
    preds.write_text("\n".join(lines) + "\n")
    info = classify_predictions(preds)
    assert info["max_trajectories_per_sample"] == 4
    assert info["pass_4_available"] is True


def test_submit_command_warns_when_passk_unavailable(tmp_path, monkeypatch):
    preds = tmp_path / "preds.jsonl"
    preds.write_text(json.dumps({"sample_id": "s1", "trajectory": []}) + "\n")
    captured = {}

    def fake_post(*, predictions, model_name, endpoint, token):
        captured["model_name"] = model_name
        return {"run_id": "run_abc"}

    monkeypatch.setattr("spar.leaderboard.submit._post_submission", fake_post)
    result = CliRunner().invoke(app, ["--predictions", str(preds), "--model-name", "m"])
    assert result.exit_code == 0, result.output
    assert captured["model_name"] == "m"
    assert "pass_4" in result.output and "null" in result.output
    assert "run_abc" in result.output


def test_submit_has_no_agent_import_flag():
    # C5: there is NO --agent import path on the submit client. Passing one is rejected as an
    # unknown option (functional check — avoids parsing Rich-rendered help, which wraps/omits
    # options differently across terminal widths + Typer/Rich versions and is flaky in CI).
    # That `--predictions` IS a real option is proven by test_submit_command_warns_* above.
    result = CliRunner().invoke(app, [
        "--agent", "spar.agents.base:AbortAgent", "--predictions", "/tmp/x.jsonl",
    ])
    assert result.exit_code != 0  # the in-process agent-import flag does not exist


def test_quota_makes_a_real_server_call_not_an_echo_stub(monkeypatch):
    monkeypatch.setattr("spar.leaderboard.submit._get_quota",
                        lambda *, endpoint, token: {"remaining": 7, "limit": 10})
    result = CliRunner().invoke(app, ["quota"])
    assert result.exit_code == 0, result.output
    assert "7" in result.output and "10" in result.output


def test_status_makes_a_real_server_call_not_an_echo_stub(monkeypatch):
    monkeypatch.setattr("spar.leaderboard.submit._get_status",
                        lambda *, run_id, endpoint, token: {"run_id": run_id, "state": "graded"})
    result = CliRunner().invoke(app, ["status", "--run-id", "run_abc"])
    assert result.exit_code == 0, result.output
    assert "run_abc" in result.output and "graded" in result.output
