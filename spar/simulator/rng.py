"""Deterministic seeded randomness with named sub-streams (module 10 §5).

All world randomness MUST flow through here. No `random`, no wall-clock, no global state.
The sub-stream keying is an append-only frozen enum: never renumber an existing key.
"""

from __future__ import annotations

import hashlib
from enum import IntEnum

import numpy as np


class SubStream(IntEnum):
    DECLINE = 0
    CHALLENGE = 1
    CAPTURE = 2
    DISPUTE = 3
    LATENCY = 4
    FRAUD = 5
    APPROVAL_BAND = 6   # seeded noise for the exposed observed_approval_band (M3, review F2)
    VOID = 7            # reserved — future stochastic void-failure modes
    REFUND = 8          # reserved — future stochastic refund-failure modes
    CHARGEBACK = 9      # reserved — future stochastic chargeback-failure modes


def stable_hash(text: str) -> int:
    """Process-independent 63-bit hash (NOT Python's salted hash())."""
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") & ((1 << 63) - 1)


def derive_seed(seed: int, trial_index: int) -> int:
    """Pure per-trial seed for pass^k (module 40 §3.3). Same inputs → same output."""
    return stable_hash(f"{seed}:{trial_index}")


def substream(
    sample_id: str,
    *,
    seed: int,
    trial_index: int,
    stream: SubStream,
    step: int,
) -> np.random.Generator:
    """A reproducible, independent Generator for (sample, trial, stream, step).

    `step` MUST be a STABLE transition ordinal (e.g. a per-route authorization-attempt
    counter), NOT the raw `elapsed_steps` clock (review G2): keying on the mutable clock
    means an extra illegal action or retry shifts every downstream pinned draw, so two
    agents reaching the same logical authorization via different action counts would get
    different outcomes — the non-reproducibility the spec forbids. The full per-trial seed
    is folded into the entropy hash (no 32-bit truncation — review G3) so pass^k trials do
    not collide.
    """
    root = np.random.SeedSequence(
        entropy=stable_hash(f"{sample_id}:{derive_seed(seed, trial_index)}"),
        spawn_key=(int(stream), int(step)),
    )
    return np.random.default_rng(root)
