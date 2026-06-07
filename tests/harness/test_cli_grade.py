"""`spar grade` replays a predictions.jsonl against the canonical seed (pass^1)."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from spar.harness.run_eval import app


def test_spar_grade_replays_predictions_pass1_only(tmp_path):
    preds = tmp_path / "preds.jsonl"
    trajectory = [
        {"tool": "select_route", "acquirer_id": "acq_a", "method": "visa"},
        {"tool": "submit_authorization"},
        {"tool": "capture"},
    ]
    preds.write_text(json.dumps({"sample_id": "spar_routing_0001", "trajectory": trajectory}) + "\n")

    out = tmp_path / "results.json"
    result = CliRunner().invoke(app, [
        "grade", "--predictions", str(preds), "--split", "lite", "--out", str(out),
    ])
    assert result.exit_code == 0, result.output
    data = json.loads(out.read_text())
    assert data["split"] == "lite"
    assert data["summary"]["pass_4"] is None         # static trajectory -> pass^1 only
    assert data["summary"]["pass_1"] is not None
    assert data["per_sample"][0]["final_state"] in {"SETTLED", "CLOSED"}
