"""NaiveCompleteAgent: the F1 reference adversary (module 30 §3, REVIEW F1).

It performs only the three trivial completion tools and never a non-trivial
correct action, so on a well-formed non-trap it must fail `outcome_correct`.
Used by build.py's F1 spot-check and the unit invariant test.
"""

from __future__ import annotations

from spar.simulator.contract import (
    Action, Capture, Observation, SelectRoute, SubmitAuthorization,
)

NAIVE_COMPLETE_TOOLS: frozenset[str] = frozenset(
    {"select_route", "submit_authorization", "capture"}
)


class NaiveCompleteAgent:
    """select_route -> submit_authorization -> capture, nothing non-trivial."""

    def __init__(self) -> None:
        self._step = 0

    def act(self, observation: Observation) -> Action:
        self._step += 1
        if self._step == 1 and observation.methods:
            acq = observation.methods[0]
            return SelectRoute(tool="select_route", acquirer_id=acq.acquirer_id,
                               method=acq.methods[0])
        if self._step == 2:
            return SubmitAuthorization(tool="submit_authorization")
        return Capture(tool="capture")
