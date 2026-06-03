"""Static, lazy-loadable viewer bundle. Reads runs_*/<model>/ outputs and emits: index.json
(small — drives lists), per-model manifest+summary, per-episode files fetched on demand, plus the
exported JSON Schema + generated TS types. No network, no scoring. The bundle is gitignored; the
in-code EpisodeRecord schema is the committed source of truth."""

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


def write_fixtures(*, bundle_dir: Path, max_examples: int = 6) -> list[Path]:
    """Copy a small, diverse set of real episode files into bundle/fixtures/ for frontend dev:
    prefer one trap, one malformed, and otherwise spread across axes. Best-effort; bundle-local."""
    index = json.loads((bundle_dir / "index.json").read_text())
    picks: list[tuple[str, str]] = []  # (model_id, sample_id)
    seen_axis: set[str] = set()
    have_trap = have_bad = False
    for m in index["models"]:
        for s in m["samples"]:
            sid, axis, is_trap = s["sample_id"], s.get("axis"), s.get("is_trap")
            ep = bundle_dir / m["id"] / "episodes" / f"{sid}.jsonl"
            if not ep.is_file():
                continue
            status = json.loads(ep.read_text().splitlines()[0]).get("status")
            want = False
            if is_trap and not have_trap:
                want, have_trap = True, True
            elif status == "malformed_action" and not have_bad:
                want, have_bad = True, True
            elif axis not in seen_axis:
                want = True
                seen_axis.add(axis)
            if want and len(picks) < max_examples:
                picks.append((m["id"], sid))
    fdir = bundle_dir / "fixtures"
    fdir.mkdir(exist_ok=True)
    out: list[Path] = []
    fixture_index = []
    for mid, sid in picks:
        src = bundle_dir / mid / "episodes" / f"{sid}.jsonl"
        dst = fdir / f"{sid}.jsonl"
        shutil.copy(src, dst)
        out.append(dst)
        fixture_index.append({"model": mid, "sample_id": sid})
    (fdir / "index.json").write_text(json.dumps({"fixtures": fixture_index}, indent=2))
    return out


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
    try:
        write_fixtures(bundle_dir=out_dir)
    except (KeyError, FileNotFoundError, IndexError):
        pass
