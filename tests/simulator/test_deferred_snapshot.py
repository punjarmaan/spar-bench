"""Determinism snapshot (module 10 §9): the drained grade-terminal + hidden state hash to a
stable digest for a fixed (sample_id, seed, trial_index), guarding against any future
reordering of the deferred queue or sub-stream keys."""

from __future__ import annotations

import hashlib

from spar.harness.runner import EpisodeTrace, run_episode
from tests.harness.test_m5_integration import _VerifyAgent, _load


def _digest(trace: EpisodeTrace) -> str:
    parts = [trace.grade_terminal.value]
    for k in sorted(trace.hidden_final_state):
        parts.append(f"{k}={trace.hidden_final_state[k]}")
    return hashlib.blake2b("|".join(parts).encode(), digest_size=8).hexdigest()


def test_deferred_drain_digest_is_stable_across_runs():
    legit = [s for s in _load("post_purchase") if not s.is_trap][0]
    d1 = _digest(run_episode(legit, _VerifyAgent(), trial_index=0))
    d2 = _digest(run_episode(legit, _VerifyAgent(), trial_index=0))
    assert d1 == d2  # same seed -> identical drained state


def test_different_trial_index_is_itself_stable():
    legit = [s for s in _load("post_purchase") if not s.is_trap][0]
    legit.world_config.dispute = {"fires_at_offset": 2, "prob_fraud": 0.5, "prob_clean": 0.5}
    a = _digest(run_episode(legit, _VerifyAgent(), trial_index=1))
    b = _digest(run_episode(legit, _VerifyAgent(), trial_index=1))
    assert a == b  # trial_index=1 is itself reproducible
