"""Load a split's JSONL (module 30 §5, F13, F17).

Published splits are immutable, hash-pinned artifacts: this loader READS them, never
regenerates. PUBLIC splits (`lite`/`main`/`diamond`) are PROJECTED rows (no gold / hidden
world_config, F17) — `load_public_split` returns them as validated dicts. The `private`
split is the full graded `Sample` and is loaded by `load_split(..., base_dir=...)`. With no
base_dir, `load_split` falls back to the bundled toy lite split (M1 behavior) so the local
`spar run`/`grade` CLI keeps working. `load_gold` (the hand-authored backbone, owned by M2)
is preserved here — many milestones import it.
"""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import Any

from spar.simulator.enums import Axis
from spar.simulator.schemas import Sample

_PUBLIC_SPLITS = {"lite", "main", "diamond"}
_ALL_SPLITS = {"lite", "main", "diamond", "private"}

# Keys a projected public row is allowed to carry (mirrors projection.public_view).
_PUBLIC_KEYS = frozenset(
    {"sample_id", "axis", "difficulty", "policy_id", "canary", "seed",
     "mandate", "cart", "methods"}
)


def _parse_full(text: str) -> list[Sample]:
    return [Sample.model_validate_json(line) for line in text.splitlines() if line.strip()]


def load_public_split(split: str, *, base_dir: Path) -> list[dict[str, Any]]:
    """Load a frozen PUBLIC (projected) split as validated dict rows (F17)."""
    if split not in _PUBLIC_SPLITS:
        raise ValueError(f"unknown public split: {split!r}")
    rows: list[dict[str, Any]] = []
    text = (Path(base_dir) / f"{split}.jsonl").read_text(encoding="utf-8")
    for line in text.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        extra = set(row) - _PUBLIC_KEYS
        if extra:  # a leaked gold/world_config key would land here
            raise ValueError(f"public split {split!r} row has non-public keys: {extra}")
        rows.append(row)
    return rows


def load_split(split: str, *, base_dir: Path | None = None) -> list[Sample]:
    """Load the FULL graded `private` split (or the bundled toy lite split when no base_dir
    is given). Public splits are projected — use `load_public_split` instead."""
    if split not in _ALL_SPLITS:
        raise ValueError(f"unknown split: {split!r}")
    if base_dir is not None:
        if split != "private":
            raise ValueError(
                f"{split!r} is published as a PROJECTED public split; "
                "use load_public_split (gold is held server-side only, F17)"
            )
        return _parse_full((Path(base_dir) / "private.jsonl").read_text(encoding="utf-8"))
    # No base_dir: the bundled toy lite split (M1 behavior; the local CLI default).
    text = resources.files("spar.dataset.toy").joinpath("lite.jsonl").read_text(encoding="utf-8")
    return _parse_full(text)


def load_gold(axis: Axis | str) -> list[Sample]:
    """Load the hand-authored gold samples for an axis from spar/dataset/gold/<axis>.jsonl.

    Registry-owned by M2; M3-M7 import this (never redefine it). Accepts an `Axis` enum or
    its string value; returns [] if no gold file is shipped for that axis.
    """
    name = axis.value if isinstance(axis, Axis) else str(axis)
    resource = resources.files("spar.dataset.gold").joinpath(f"{name}.jsonl")
    if not resource.is_file():
        return []
    return _parse_full(resource.read_text(encoding="utf-8"))
