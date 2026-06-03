"""Canary minting + stamping.

Split assignment is owned by `plan.py` (the single mechanism) + the hand-authored
backbone — NOT here; there is deliberately no `assign_split`. This module mints the
contamination canary: a FRESH random `spar:<uuid4>` per build (recorded in the
manifest/HF card). It is fresh — never a function of `build_seed` — so a leaked seed
cannot be used to pre-compute and scrub the canary. Sample assignment stays
deterministic; only the canary varies per build.
"""

from __future__ import annotations

import uuid

from spar.simulator.schemas import Sample


def make_canary() -> str:
    """Mint a fresh random `spar:<uuid4>` for this build (recorded in the manifest)."""
    return f"spar:{uuid.uuid4()}"


def apply_canary(sample: Sample, canary: str) -> Sample:
    """Return a copy of `sample` with the build canary stamped on it."""
    return sample.model_copy(update={"canary": canary})
