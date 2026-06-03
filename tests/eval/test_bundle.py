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
                         "score": 0.5, "model_graded": False}]}))
    (model_dir / "trajectories" / "s1.jsonl").write_text(
        json.dumps({"schema_version": 1, "sample_id": "s1", "trial_index": 0, "turns": []}) + "\n")


def test_build_bundle_emits_lazy_layout(tmp_path: Path):
    runs = tmp_path / "runs"; runs.mkdir()
    _seed_run(runs)
    out = tmp_path / "bundle"
    build_bundle(runs_dir=runs, out_dir=out)
    index = json.loads((out / "index.json").read_text())
    assert index["models"][0]["id"] == "test-model"
    assert index["models"][0]["samples"][0]["sample_id"] == "s1"
    assert "turns" not in json.dumps(index)            # lazy: no episode payload in index
    assert (out / "test-model" / "episodes" / "s1.jsonl").is_file()
    assert (out / "schema" / "trajectory.schema.json").is_file()
    assert (out / "types" / "spar.d.ts").is_file()


def test_schema_export_and_ts_cover_core_fields(tmp_path: Path):
    schema = export_schema()
    assert "sample_id" in schema["properties"]
    ts = schema_to_typescript(schema)
    assert "sample_id" in ts and "interface" in ts
