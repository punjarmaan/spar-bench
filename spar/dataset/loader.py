"""Load a split's JSONL into validated Sample objects. M1 ships a one-sample toy split."""

from __future__ import annotations

from importlib import resources
from pathlib import Path

from spar.simulator.schemas import Sample

_TOY_SPLITS = {"lite", "main", "diamond"}


def load_split(split: str, *, base_dir: Path | None = None) -> list[Sample]:
    if split not in _TOY_SPLITS:
        raise ValueError(f"unknown split: {split!r}")
    if base_dir is not None:
        # M7 frozen-artifact path: load <base_dir>/<split>.jsonl.
        text = (base_dir / f"{split}.jsonl").read_text(encoding="utf-8")
    else:
        # M1 only bundles the toy lite split; M7 replaces this with built JSONL artifacts.
        text = resources.files("spar.dataset.toy").joinpath("lite.jsonl").read_text(encoding="utf-8")
    return [Sample.model_validate_json(line) for line in text.splitlines() if line.strip()]
