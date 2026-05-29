import json

from typer.testing import CliRunner

from spar.harness.run_eval import app


def test_spar_run_end_to_end_writes_results(tmp_path):
    out = tmp_path / "results.json"
    runner = CliRunner()
    result = runner.invoke(app, [
        "run", "--split", "lite",
        "--agent", "spar.agents.reference_agents:HappyPathAgent",
        "--out", str(out),
    ])
    assert result.exit_code == 0, result.output
    data = json.loads(out.read_text())
    assert data["split"] == "lite"
    assert data["summary"]["n_samples"] >= 1
    # HappyPathAgent should settle the toy sample → CLOSED-equivalent outcome credited.
    assert data["per_sample"][0]["final_state"] in {"SETTLED", "CLOSED"}
