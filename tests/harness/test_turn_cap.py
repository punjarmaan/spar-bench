"""Hard per-episode turn-cap backstop (live-spend safety).

`World.step` returns a VERBATIM ledger replay BEFORE incrementing `elapsed_steps` whenever a
keyed mutating action reuses an idempotency_key already in the ledger (world.py ~line 618-624).
So an agent that repeatedly emits the SAME keyed action makes no FSM progress and freezes
`elapsed_steps` — the runner's `elapsed_steps >= max_steps` step-budget break can NEVER fire,
and the loop spins forever (for a live model: unbounded paid completions). `run_episode` must
carry a hard turn counter, INDEPENDENT of `elapsed_steps`, that guarantees termination.
"""

from __future__ import annotations

from spar.harness.runner import _HARD_TURN_CAP_FACTOR, run_episode
from spar.simulator.contract import (
    Action,
    Observation,
    SelectRoute,
    SubmitAuthorization,
)
from spar.simulator.enums import FsmState
from spar.simulator.world import World
from tests.eval._fakes import abort_sample

_KEY = "loop"


class _LoopingAgent:
    """Drives a TRUE no-progress keyed-replay loop.

    Turn 1: SelectRoute (advances the FSM).
    Turn 2: SubmitAuthorization(idempotency_key="loop") — advances AND ledgers the key.
    Turn 3+: the SAME SubmitAuthorization(idempotency_key="loop") FOREVER — each is a verbatim
    ledger replay, so `World.step` returns before `elapsed_steps += 1`. `elapsed_steps` is frozen
    at 2 and never reaches max_steps; only the hard turn cap can break the loop.
    """

    def __init__(self) -> None:
        self._selected = False

    def act(self, observation: Observation) -> Action:
        if not self._selected:
            self._selected = True
            m = observation.methods[0]
            return SelectRoute(
                tool="select_route", acquirer_id=m.acquirer_id, method=m.methods[0]
            )
        return SubmitAuthorization(tool="submit_authorization", idempotency_key=_KEY)


def test_looping_agent_is_a_true_no_progress_loop() -> None:
    """Prove the loop is REAL: once the keyed auth is ledgered, replaying it freezes
    `elapsed_steps`. Driving World.step directly (no runner guard) shows the clock never
    advances no matter how many identical keyed authorizations we feed it."""
    sample = abort_sample()
    world = World(sample, trial_index=0)
    world.reset()
    agent = _LoopingAgent()
    # Turn 1: select_route (advances).
    world.step(agent.act(world.observe()))
    # Turn 2: the keyed auth advances AND records the key in the ledger.
    world.step(agent.act(world.observe()))
    frozen = world.elapsed_steps
    assert frozen > 0
    # Turns 3..N: the SAME keyed auth — a verbatim replay; elapsed_steps never moves.
    for _ in range(50):
        world.step(agent.act(world.observe()))
        assert world.elapsed_steps == frozen, "keyed replay must not advance elapsed_steps"
    # The step-budget guard keys on elapsed_steps, which is frozen below max_steps forever:
    assert frozen < sample.world_config.max_steps


def test_turn_cap_terminates_the_loop() -> None:
    """`run_episode` against the no-progress looping agent RETURNS (does not hang) and aborts
    via the hard turn-cap backstop — a non-deliberate abort (made_decision stays False)."""
    sample = abort_sample()
    trace = run_episode(sample, _LoopingAgent())
    assert trace.final_state == FsmState.ABORTED
    assert trace.abort_reason == "turn_cap_exhausted"
    assert trace.terminating_action == "turn_cap_exhausted"
    assert trace.made_decision is False
    # The cap is elapsed_steps-INDEPENDENT: it must be the larger backstop, > max_steps.
    assert _HARD_TURN_CAP_FACTOR >= 1
    assert len(trace.action_log) <= sample.world_config.max_steps * _HARD_TURN_CAP_FACTOR + 1
