"""spar bundle: static lazy viewer bundle from run outputs (enrichment Phase B / T7)."""
import json
from pathlib import Path

from spar.eval.bundle import build_bundle, export_schema, schema_to_typescript


def _seed_run(root: Path):
    model_dir = root / "test-model"
    (model_dir / "trajectories").mkdir(parents=True)
    (model_dir / "run_manifest.json").write_text(json.dumps(
        {"model": "test-model", "class": "open", "canary": "spar:abc",
         "grader_model": "g", "responder_model": "r"}))
    (model_dir / "main.results.json").write_text(json.dumps(
        {"split": "main", "summary": {"pass_1": 0.5, "n_samples": 1},
         "per_sample": [{"sample_id": "s1", "axis": "routing", "is_trap": False,
                         "score": 0.5, "model_graded": False, "intent_spec": "explicit",
                         "final_state": "CLOSED", "trials_n": 1}]}))
    (model_dir / "trajectories" / "s1.jsonl").write_text(
        json.dumps({"schema_version": 1, "sample_id": "s1", "trial_index": 0, "turns": []}) + "\n")


def test_build_bundle_emits_lazy_layout(tmp_path: Path):
    runs = tmp_path / "runs"
    runs.mkdir()
    _seed_run(runs)
    out = tmp_path / "bundle"
    build_bundle(runs_dir=runs, out_dir=out, spar_version="1.0.0")
    index = json.loads((out / "1.0.0" / "index.json").read_text())
    assert index["models"][0]["id"] == "test-model"
    assert index["models"][0]["samples"][0]["sample_id"] == "s1"
    assert "turns" not in json.dumps(index)            # lazy: no episode payload in index
    assert (out / "1.0.0" / "test-model" / "episodes" / "s1.jsonl").is_file()
    assert (out / "1.0.0" / "schema" / "trajectory.schema.json").is_file()
    assert (out / "1.0.0" / "types" / "spar.d.ts").is_file()


def test_schema_export_and_ts_cover_core_fields(tmp_path: Path):
    schema = export_schema()
    assert "sample_id" in schema["properties"]
    ts = schema_to_typescript(schema)
    assert "sample_id" in ts and "interface" in ts


def test_index_rows_carry_filter_fields_across_splits(tmp_path: Path):
    runs = tmp_path / "runs"
    runs.mkdir()
    _seed_run(runs)
    (runs / "test-model" / "redline.results.json").write_text(json.dumps(
        {"split": "redline", "summary": {"n_samples": 1},
         "per_sample": [{"sample_id": "t1", "axis": "routing", "is_trap": True, "score": 0.0,
                         "model_graded": False, "intent_spec": "semantic",
                         "final_state": "ESCALATED", "trials_n": 4}]}))
    out = tmp_path / "bundle"
    build_bundle(runs_dir=runs, out_dir=out, spar_version="1.0.0")
    samples = json.loads((out / "1.0.0" / "index.json").read_text())["models"][0]["samples"]
    by_split = {s["split"]: s for s in samples}
    assert by_split["main"]["intent_spec"] == "explicit" and by_split["main"]["trials_n"] == 1
    assert by_split["main"]["final_state"] == "CLOSED"
    assert by_split["redline"]["is_trap"] is True and by_split["redline"]["intent_spec"] == "semantic"


def test_canary_is_scrubbed_everywhere(tmp_path: Path):
    runs = tmp_path / "runs"
    runs.mkdir()
    _seed_run(runs)  # manifest canary = "spar:abc"
    # defensively plant the canary in an episode too (future runs might embed it)
    (runs / "test-model" / "trajectories" / "s1.jsonl").write_text(json.dumps(
        {"schema_version": 1, "sample_id": "s1", "trial_index": 0,
         "system_prompt": "spar:abc do not train", "turns": []}) + "\n")
    out = tmp_path / "bundle"
    build_bundle(runs_dir=runs, out_dir=out, spar_version="1.0.0")
    blob = "".join(p.read_text() for p in out.rglob("*.json")) + \
           "".join(p.read_text() for p in out.rglob("*.jsonl"))
    assert "spar:abc" not in blob              # gone from manifest, results, episodes, index
    assert "spar:REDACTED-CANARY" in blob      # placeholder present


def test_bundle_manifest_and_subset(tmp_path: Path):
    runs = tmp_path / "runs"
    runs.mkdir()
    _seed_run(runs)
    out = tmp_path / "bundle"
    build_bundle(runs_dir=runs, out_dir=out, spar_version="1.0.0")
    man = json.loads((out / "1.0.0" / "bundle_manifest.json").read_text())
    assert man["spar_version"] == "1.0.0" and man["schema_version"] == 1
    assert man["model_count"] == 1 and man["sample_count"] == 1
    assert man["canary_scrubbed"] is True
    out2 = tmp_path / "bundle2"
    build_bundle(runs_dir=runs, out_dir=out2, spar_version="1.0.0", models=["nope"])
    assert json.loads((out2 / "1.0.0" / "index.json").read_text())["models"] == []


def test_bundle_cli_writes_versioned_dir(tmp_path: Path):
    from typer.testing import CliRunner

    from spar.harness.run_eval import app

    runs = tmp_path / "runs"
    runs.mkdir()
    _seed_run(runs)
    out = tmp_path / "bundle"
    result = CliRunner().invoke(
        app, ["bundle", "--runs", str(runs), "--out", str(out), "--version", "1.0.0"]
    )
    assert result.exit_code == 0, result.output
    assert (out / "1.0.0" / "index.json").is_file()
    assert (out / "1.0.0" / "bundle_manifest.json").is_file()
