"""Static, lazy-loadable viewer bundle. Reads runs_*/<model>/ outputs and emits: index.json
(small — drives lists), samples.json (sample-centric rows embedding every model's run summary),
per-model manifest+summary, per-episode files fetched on demand, plus the
exported JSON Schema + generated TS types. No network, no scoring. The bundle is gitignored; the
in-code EpisodeRecord schema is the committed source of truth."""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from spar.eval.trajectory import EpisodeRecord, TRAJECTORY_SCHEMA_VERSION


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
                "intent_spec": ps.get("intent_spec"),
                "final_state": ps.get("final_state"),
                "trials_n": ps.get("trials_n"),
            })
    return {
        "id": manifest.get("model"), "class": manifest.get("class"),
        "canary": manifest.get("canary"),
        "samples": samples,
    }


def _load_difficulties(dataset_dir: Path | None) -> dict[str, str]:
    """sample_id -> difficulty, from the built dataset's *.jsonl files. Empty when no dataset
    is given (difficulty then emits as null; the viewer renders an em dash)."""
    if dataset_dir is None:
        return {}
    out: dict[str, str] = {}
    for f in sorted(dataset_dir.glob("*.jsonl")):
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            if d.get("sample_id") and d.get("difficulty"):
                out[d["sample_id"]] = d["difficulty"]
    return out


def _sample_rows(model_dirs: list[Path], difficulties: dict[str, str]) -> list[dict[str, Any]]:
    """samples.json rows: one per sample_id, embedding every model's run summary. Redline trial
    fields (trials_c/trials_safe_c/pass_4_safety) are nulled on main rows — per-sample pass^k
    only means something on the multi-trial redline split."""
    by_sample: dict[str, dict[str, Any]] = {}
    for model_dir in model_dirs:
        model_id = json.loads((model_dir / "run_manifest.json").read_text()).get("model")
        for results in sorted(model_dir.glob("*.results.json")):
            data = json.loads(results.read_text())
            split = data.get("split")
            redline = split == "redline"
            for ps in data.get("per_sample", []):
                sid = ps["sample_id"]
                # sample-level fields take the first model's values ("first model wins")
                row = by_sample.setdefault(sid, {
                    "sample_id": sid, "axis": ps.get("axis"), "split": split,
                    "intent_spec": ps.get("intent_spec"), "is_trap": ps.get("is_trap"),
                    "difficulty": difficulties.get(sid), "trials_n": ps.get("trials_n"),
                    "runs": [],
                })
                row["runs"].append({
                    "model": model_id,
                    "score": ps.get("score"),
                    "final_state": ps.get("final_state"),
                    "trials_n": ps.get("trials_n"),
                    "trials_c": ps.get("trials_c") if redline else None,
                    "trials_safe_c": ps.get("trials_safe_c") if redline else None,
                    "pass_4_safety": ps.get("pass_4_safety") if redline else None,
                    "has_trajectory":
                        (model_dir / "trajectories" / f"{sid}.jsonl").is_file(),
                })
    for row in by_sample.values():
        row["runs"].sort(key=lambda r: r["model"])
    return list(by_sample.values())


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


CANARY_PLACEHOLDER = "spar:REDACTED-CANARY"


def _scrub(text: str, canaries: Iterable[str]) -> str:
    """Replace every real canary occurrence with the public placeholder."""
    for c in canaries:
        if c:
            text = text.replace(c, CANARY_PLACEHOLDER)
    return text


def _write_scrubbed(src: Path, dst: Path, canaries: Iterable[str]) -> None:
    dst.write_text(_scrub(src.read_text(), canaries))


def build_bundle(*, runs_dir: Path, out_dir: Path, spar_version: str,
                 models: list[str] | None = None, dataset_dir: Path | None = None) -> None:
    out_dir = out_dir / spar_version
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "schema").mkdir(exist_ok=True)
    (out_dir / "types").mkdir(exist_ok=True)
    schema = export_schema()
    (out_dir / "schema" / "trajectory.schema.json").write_text(json.dumps(schema, indent=2))
    (out_dir / "types" / "spar.d.ts").write_text(schema_to_typescript(schema))

    model_dirs = sorted(p for p in runs_dir.iterdir() if (p / "run_manifest.json").is_file())
    if models is not None:
        wanted = set(models)
        model_dirs = [p for p in model_dirs if p.name in wanted]

    canaries = {
        json.loads((d / "run_manifest.json").read_text()).get("canary") for d in model_dirs
    }
    canaries.discard(None)

    model_rows: list[dict[str, Any]] = []
    for model_dir in model_dirs:
        row = _index_rows(model_dir)
        if row.get("canary"):
            row["canary"] = CANARY_PLACEHOLDER
        model_rows.append(row)
        dest = out_dir / model_dir.name
        (dest / "episodes").mkdir(parents=True, exist_ok=True)
        # Keep the per-model copy named run_manifest.json: it's the verbatim (canary-scrubbed) raw
        # run manifest, NOT the authoritative bundle metadata — that lives in the top-level
        # bundle_manifest.json (schema_version, canary_scrubbed, model_count, …).
        _write_scrubbed(model_dir / "run_manifest.json", dest / "run_manifest.json", canaries)
        for results in model_dir.glob("*.results.json"):
            _write_scrubbed(results, dest / results.name, canaries)
        traj_dir = model_dir / "trajectories"
        if traj_dir.is_dir():
            for traj in traj_dir.glob("*.jsonl"):
                _write_scrubbed(traj, dest / "episodes" / traj.name, canaries)
    (out_dir / "index.json").write_text(_scrub(json.dumps({"models": model_rows}, indent=2), canaries))
    (out_dir / "samples.json").write_text(_scrub(json.dumps({
        "schema_version": TRAJECTORY_SCHEMA_VERSION,
        "spar_version": spar_version,
        "samples": _sample_rows(model_dirs, _load_difficulties(dataset_dir)),
    }, indent=2), canaries))
    (out_dir / "bundle_manifest.json").write_text(json.dumps({
        "schema_version": TRAJECTORY_SCHEMA_VERSION,
        "spar_version": spar_version,
        "model_count": len(model_rows),
        "sample_count": sum(len(m["samples"]) for m in model_rows),
        "canary_scrubbed": True,
        "canary_placeholder": CANARY_PLACEHOLDER,
    }, indent=2))
    try:
        write_fixtures(bundle_dir=out_dir)
    except (KeyError, FileNotFoundError, IndexError):
        pass
