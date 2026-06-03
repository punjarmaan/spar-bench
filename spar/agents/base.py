"""The Agent protocol and a trivial reference agent for smoke tests."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from spar.simulator.contract import Abort, Action, Observation


@runtime_checkable
class Agent(Protocol):
    def act(self, observation: Observation) -> Action: ...


class AbortAgent:
    """Always aborts immediately. Smoke-tests the loop."""

    def act(self, observation: Observation) -> Action:
        return Abort(tool="abort", reason="noop-baseline")
