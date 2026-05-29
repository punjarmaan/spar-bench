"""Per-split manifest builder (module 30 §5). The manifest is consumed by the
module-50 validation tests and pins each split by `sample_ids_sha256` (F13).
"""

from __future__ import annotations

import hashlib

from spar.simulator.enums import Axis, Difficulty
from spar.simulator.schemas import Sample

SCHEMA_VERSION = 1


def _empty_counts() -> dict[str, dict[str, dict[str, int]]]:
    return {
        axis.value: {d.value: {"trap": 0, "non_trap": 0} for d in Difficulty}
        for axis in Axis
    }


def _sample_ids_sha256(samples: list[Sample]) -> str:
    joined = "\n".join(sorted(s.sample_id for s in samples))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def build_manifest(samples: list[Sample], *, split: str, build_seed: int,
                   canary: str, spar_version: str) -> dict[str, object]:
    """Build the manifest.json dict for one split."""
    counts = _empty_counts()
    n_traps = 0
    for s in samples:
        cell = "trap" if s.is_trap else "non_trap"
        counts[s.axis.value][s.difficulty.value][cell] += 1
        if s.is_trap:
            n_traps += 1
    n = len(samples)
    return {
        "split": split,
        "spar_version": spar_version,
        "schema_version": SCHEMA_VERSION,
        "build_seed": build_seed,
        "canary": canary,
        "n_samples": n,
        "counts": counts,
        "n_diamond": sum(1 for s in samples if s.diamond),
        "n_model_graded": sum(1 for s in samples if s.model_graded),
        "trap_fraction": (n_traps / n) if n else 0.0,
        "sample_ids_sha256": _sample_ids_sha256(samples),
    }
