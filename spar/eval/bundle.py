"""Static, lazy-loadable viewer bundle (enrichment spec §7). Reads runs_*/<model>/ outputs and
emits: index.json (small — drives lists), per-model manifest+summary, per-episode files fetched on
demand, plus the exported JSON Schema + generated TS types. No network, no scoring. The bundle is
gitignored; the in-code EpisodeRecord schema is the committed source of truth."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from spar.eval.trajectory import EpisodeRecord


def export_schema() -> dict[str, Any]:
    return EpisodeRecord.model_json_schema()


def schema_to_typescript(schema: dict[str, Any]) -> str:
    """Minimal JSON-Schema -> TS interface emitter for our own models (objects/arrays/primitives/
    optionals/$defs). Sufficient for EpisodeRecord; not a general converter."""
    defs = schema.get("$defs", {})
    out: list[str] = ["// AUTO-GENERATED from EpisodeRecord.model_json_schema(). Do not edit by hand."]

    def ts_type(node: dict[str, Any]) -> str:
        if "$ref" in node:
            return node["$ref"].split("/")[-1]
        if "anyOf" in node:
            return " | ".join(ts_type(s) for s in node["anyOf"])
        t = node.get("type")
        if t == "array":
            return f"{ts_type(node.get('items', {}))}[]"
        if t == "object":
            return "Record<string, unknown>"
        return {"string": "string", "integer": "number", "number": "number",
                "boolean": "boolean", "null": "null"}.get(t, "unknown")

    def emit(name: str, node: dict[str, Any]) -> None:
        req = set(node.get("required", []))
        out.append(f"export interface {name} {{")
        for fname, fnode in node.get("properties", {}).items():
            opt = "" if fname in req else "?"
            out.append(f"  {fname}{opt}: {ts_type(fnode)};")
        out.append("}")

    for dname, dnode in defs.items():
        emit(dname, dnode)
    emit("EpisodeRecord", schema)
    return "\n".join(out) + "\n"


def _index_rows(model_dir: Path) -> dict[str, Any]:
    manifest = json.loads((model_dir / "run_manifest.json").read_text())
    samples: list[dict[str, Any]] = []
    for results in sorted(model_dir.glob("*.results.json")):
        data = json.loads(results.read_text())
        for ps in data.get("per_sample", []):
            samples.append({
                "sample_id": ps["sample_id"], "axis": ps.get("axis"),
                "is_trap": ps.get("is_trap"), "score": ps.get("score"),
                "split": data.get("split"),
            })
    return {
        "id": manifest.get("model"), "class": manifest.get("class"),
        "canary": manifest.get("canary"),
        "samples": samples,
    }


def build_bundle(*, runs_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "schema").mkdir(exist_ok=True)
    (out_dir / "types").mkdir(exist_ok=True)
    schema = export_schema()
    (out_dir / "schema" / "trajectory.schema.json").write_text(json.dumps(schema, indent=2))
    (out_dir / "types" / "spar.d.ts").write_text(schema_to_typescript(schema))

    models: list[dict[str, Any]] = []
    for model_dir in sorted(p for p in runs_dir.iterdir() if (p / "run_manifest.json").is_file()):
        row = _index_rows(model_dir)
        models.append(row)
        dest = out_dir / model_dir.name
        (dest / "episodes").mkdir(parents=True, exist_ok=True)
        shutil.copy(model_dir / "run_manifest.json", dest / "manifest.json")
        for results in model_dir.glob("*.results.json"):
            shutil.copy(results, dest / results.name)
        traj_dir = model_dir / "trajectories"
        if traj_dir.is_dir():
            for traj in traj_dir.glob("*.jsonl"):
                shutil.copy(traj, dest / "episodes" / traj.name)
    (out_dir / "index.json").write_text(json.dumps({"models": models}, indent=2))
