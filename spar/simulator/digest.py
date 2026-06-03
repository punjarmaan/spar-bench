"""Deterministic trajectory digest for the FSM determinism snapshot.

Hashes the full (state, action, response) trajectory of a World driven by a fixed action
list into a stable hex digest. Uses blake2b (process-independent), never Python's salted hash.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

from spar.simulator.contract import Action
from spar.simulator.world import World


def trajectory_digest(world: World, trajectory: Sequence[Action]) -> str:
    """Drive `world` through `trajectory`, hashing each (state, action, response) triple."""
    hasher = hashlib.blake2b(digest_size=16)
    hasher.update(world.state.value.encode("utf-8"))
    for action in trajectory:
        if world.is_agent_terminal():
            break
        resp = world.step(action)
        triple = (
            f"{world.state.value}|{action.model_dump_json()}|"
            f"{resp.status.value}:{resp.reason_code}:{resp.challenge_token}"
        )
        hasher.update(triple.encode("utf-8"))
    return hasher.hexdigest()
