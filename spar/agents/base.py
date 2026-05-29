"""The Agent protocol (module 20 §5) and a trivial reference agent for M1 smoke tests."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from spar.simulator.contract import Abort, Action, Observation


@runtime_checkable
class Agent(Protocol):
    def act(self, observation: Observation) -> Action: ...


class AbortAgent:
    """Trivial baseline: always aborts immediately. Used only to smoke-test the loop."""

    def act(self, observation: Observation) -> Action:
        return Abort(tool="abort", reason="noop-baseline")
