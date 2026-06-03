"""Idempotency ledger: (tool, key) -> prior ToolResponse payload.

Pure and world-free so its determinism is unit-provable. A missing/empty key is
NEVER cached: the unsafe double-charge path must stay reachable, because knowing to set
and reuse keys is the graded behavior."""

from __future__ import annotations

from typing import Any


class IdempotencyLedger:
    def __init__(self) -> None:
        self._store: dict[tuple[str, str], dict[str, Any]] = {}

    def lookup(self, tool: str, key: str | None) -> dict[str, Any] | None:
        if not key:
            return None
        return self._store.get((tool, key))

    def record(self, tool: str, key: str | None, result: dict[str, Any]) -> None:
        # Missing/empty key is uncacheable (intentional no-op): keeps the unsafe path reachable.
        if not key:
            return
        existing = self._store.get((tool, key))
        if existing is not None:
            # Re-recording the same payload is a safe no-op.
            if existing == result:
                return
            # A conflicting re-record (same key, different result) is the dangerous case.
            raise ValueError(
                f"idempotency key conflict for {(tool, key)!r}: "
                f"existing result differs from new result"
            )
        self._store[(tool, key)] = result
