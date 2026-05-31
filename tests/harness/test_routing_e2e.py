from spar.dataset.loader import load_gold
from spar.harness.graders import score
from spar.harness.runner import run_episode
from spar.simulator.contract import (
    Action, Capture, Observation, Retry, SelectRoute, SubmitAuthorization,
)
from spar.simulator.enums import FsmState


def _medium_gold():
    return {s.sample_id: s for s in load_gold("routing")}["spar_routing_gold_0002"]


def _by_band(methods):
    rank = {"high": 2, "med": 1, "low": 0}
    return max(methods, key=lambda m: rank[m.observed_approval_band])


def _cheapest(methods):
    return min(methods, key=lambda m: m.advertised_fee_bps)


class _RoutingAgent:
    """select(pick) -> submit -> (retry-same on decline)* -> capture. Tracks its own phase."""

    def __init__(self, pick) -> None:
        self._pick = pick
        self._selected = False
        self._submitted_once = False
        self._tries = 0

    def act(self, obs: Observation) -> Action:
        if not self._selected:
            self._selected = True
            chosen = self._pick(obs.methods)
            return SelectRoute(tool="select_route", acquirer_id=chosen.acquirer_id,
                               method=chosen.methods[0])
        last = obs.context.last_reason_code
        if last is not None and self._tries < 8:
            self._tries += 1
            self._submitted_once = False
            return Retry(tool="retry", strategy="same")
        if not self._submitted_once:
            self._submitted_once = True
            return SubmitAuthorization(tool="submit_authorization")
        return Capture(tool="capture")


def test_oracle_route_agent_scores_one_on_routing_gold():
    sample = _medium_gold()
    trace = run_episode(sample, _RoutingAgent(_by_band), trial_index=0)
    assert trace.completed_route_id == sample.gold.oracle_route
    assert trace.final_state is FsmState.SETTLED
    assert trace.grade_terminal is FsmState.CLOSED  # drained
    s = score(sample, trace)
    assert s.route_score == 1.0
    assert s.score == 1.0


def test_cheapest_advertised_agent_scores_strictly_below_oracle_on_route():
    sample = _medium_gold()
    oracle_trace = run_episode(sample, _RoutingAgent(_by_band), trial_index=0)
    cheap_trace = run_episode(sample, _RoutingAgent(_cheapest), trial_index=0)
    cheapest = min(sample.world_config.acquirers, key=lambda a: a.advertised_fee_bps)
    assert cheap_trace.completed_route_id == cheapest.acquirer_id
    oracle_s = score(sample, oracle_trace)
    cheap_s = score(sample, cheap_trace)
    assert cheap_s.route_score is not None and cheap_s.route_score < 1.0
    assert cheap_s.score < oracle_s.score


def _trap_gold():
    return {s.sample_id: s for s in load_gold("routing")}["spar_routing_gold_0003"]


class _AbortOnNoSupportedRouteAgent:
    """Correct play on the no-supported-route trap: deliberately abort."""

    def act(self, obs: Observation) -> Action:
        from spar.simulator.contract import Abort
        return Abort(tool="abort", reason="no acquirer supports the mandated instrument/geo")


class _ForceUnsupportedRouteAgent:
    """The trap bait: try to complete on the unsupported route anyway (World rejects it)."""

    def __init__(self, acquirer_id: str, method: str) -> None:
        self._acquirer_id, self._method = acquirer_id, method

    def act(self, obs: Observation) -> Action:
        return SelectRoute(tool="select_route", acquirer_id=self._acquirer_id,
                           method=self._method)


def test_trap_correct_abort_is_credited_as_grade_terminal_aborted():
    sample = _trap_gold()
    trace = run_episode(sample, _AbortOnNoSupportedRouteAgent(), trial_index=0)
    assert trace.grade_terminal is FsmState.ABORTED
    assert trace.abort_reason is None        # deliberate, not step-budget (F6)
    assert trace.completed_route_id is None   # no route completed
    s = score(sample, trace)
    assert s.outcome_correct is True
    assert s.score == 1.0                     # correct trap abort -> full credit


def test_trap_unsupported_route_completion_is_blocked_and_not_credited():
    sample = _trap_gold()
    bait = sample.mandate.allowed_instruments[0]  # `amex`, unsupported everywhere
    agent = _ForceUnsupportedRouteAgent(sample.world_config.acquirers[1].acquirer_id, bait)
    trace = run_episode(sample, agent, trial_index=0)
    assert trace.completed_route_id is None             # never completed an unsupported route
    assert trace.abort_reason == "step_budget_exhausted"
    s = score(sample, trace)
    assert s.outcome_correct is False  # F6: a forced step-budget abort is not a correct abort
    assert s.score == 0.0
