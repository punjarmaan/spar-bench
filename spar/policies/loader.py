"""Load an in-context policy document by id."""

from __future__ import annotations

from importlib import resources

KNOWN_POLICIES: frozenset[str] = frozenset({"default_v1"})


def load_policy(policy_id: str) -> str:
    """Return the prose policy doc the agent receives in-context."""
    if policy_id not in KNOWN_POLICIES:
        raise ValueError(f"unknown policy_id: {policy_id!r}")
    return resources.files("spar.policies").joinpath(f"{policy_id}.md").read_text(
        encoding="utf-8"
    )
