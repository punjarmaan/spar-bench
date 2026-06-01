"""Shared helpers for the read-only sample validity audit (no dataset mutation)."""
from __future__ import annotations

import json
from pathlib import Path

from spar.dataset.loader import load_graded_split
from spar.simulator.schemas import Sample

DATASET_DIR = Path("build/ds/private")
SPLITS = ("main", "lite", "diamond")

# Guardrail (a): behavioral evidence ONLY from verified-clean runs. Mining the wall-contaminated
# runs_main/deepseek-v3.2 (111 malformed 403s) yields FALSE sample defects (the stale_state trap).
ADMITTED_RUNS = [
    {"model": "gemini-2.0-flash",  "dir": "runs_main/gemini-2.0-flash",  "split": "main"},
    {"model": "gpt-oss-120b-free", "dir": "runs_main/gpt-oss-120b-free", "split": "main"},
    {"model": "gpt-oss-120b-free", "dir": "runs_main/gpt-oss-120b-free", "split": "lite"},
    {"model": "gpt-oss-120b-free", "dir": "runs_main/gpt-oss-120b-free", "split": "diamond"},
    {"model": "gpt-oss-120b-free", "dir": "runs_probe/gpt-oss-120b-free", "split": "probe"},
    {"model": "qwen3-235b-a22b",   "dir": "runs_probe/qwen3-235b-a22b",   "split": "probe"},
    {"model": "deepseek-v3.2",     "dir": "runs_probe/deepseek-v3.2",     "split": "probe"},
]


def load_all_samples(dataset_dir: Path = DATASET_DIR) -> list[Sample]:
    out: list[Sample] = []
    for split in SPLITS:
        out.extend(load_graded_split(split, base_dir=dataset_dir))
    return out


def cell_key(s: Sample) -> tuple[str, str, bool, str]:
    return (s.axis.value, s.difficulty.value, s.is_trap, s.intent_spec.value)


def _trajectory_status(run_dir: str, sample_id: str) -> str | None:
    p = Path(run_dir) / "trajectories" / f"{sample_id}.jsonl"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8")).get("status")


def load_clean_outcomes() -> dict[str, list[dict]]:
    """sample_id -> list of per-model clean outcomes (probe rows share main sample_ids)."""
    out: dict[str, list[dict]] = {}
    for run in ADMITTED_RUNS:
        results = Path(run["dir"]) / f"{run['split']}.results.json"
        if not results.exists():
            continue
        for ps in json.loads(results.read_text(encoding="utf-8")).get("per_sample", []):
            sid = ps["sample_id"]
            out.setdefault(sid, []).append({
                "model": run["model"], "run_dir": run["dir"],
                "status": _trajectory_status(run["dir"], sid),
                "final_state": ps.get("final_state"), "score": ps.get("score"),
                "is_trap": ps.get("is_trap"), "outcome_correct": ps.get("outcome_correct"),
                "catastrophic_class": ps.get("catastrophic_class"),
                "unsafe_completion": ps.get("unsafe_completion"),
            })
    return out
