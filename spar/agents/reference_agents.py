"""Reference baseline agents (module 20 §5; full set lands in M2/M4)."""

from __future__ import annotations

from spar.simulator.contract import (
    Action, Capture, Observation, SelectRoute, SubmitAuthorization,
)


class HappyPathAgent:
    """Selects the first route, authorizes, captures. For M1 smoke tests only."""

    def __init__(self) -> None:
        self._step = 0

    def act(self, observation: Observation) -> Action:
        self._step += 1
        if self._step == 1:
            acq = observation.methods[0]
            return SelectRoute(tool="select_route", acquirer_id=acq.acquirer_id, method=acq.methods[0])
        if self._step == 2:
            return SubmitAuthorization(tool="submit_authorization")
        return Capture(tool="capture")
