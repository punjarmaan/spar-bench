"""Deferred-event queue: a min-heap of (fire_at_step, seq, event).

Ties on `fire_at_step` break by a monotonically increasing insertion `seq`, never by
dict/set iteration order, so the drain is fully deterministic per seed.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class DeferredKind(StrEnum):
    CAPTURE_RESULT = "capture_result"
    DISPUTE_FILED = "dispute_filed"
    REVOCATION = "revocation"


@dataclass
class DeferredEvent:
    fire_at_step: int
    kind: DeferredKind
    payload: dict[str, Any] = field(default_factory=dict)


class DeferredQueue:
    """Min-heap keyed by (fire_at_step, seq). Owned by World; drained by the runner."""

    def __init__(self) -> None:
        self._heap: list[tuple[int, int, DeferredEvent]] = []
        self._seq = 0

    def push(self, event: DeferredEvent) -> None:
        heapq.heappush(self._heap, (event.fire_at_step, self._seq, event))
        self._seq += 1

    def drain_through(self, *, final_step: int) -> list[DeferredEvent]:
        """Pop and return every event with fire_at_step <= final_step, in heap order."""
        fired: list[DeferredEvent] = []
        while self._heap and self._heap[0][0] <= final_step:
            fired.append(heapq.heappop(self._heap)[2])
        return fired
