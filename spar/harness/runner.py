"""Episode runner: drives obs -> agent -> tool -> ... -> agent-terminal (module 40 §2)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from spar.agents.base import Agent
from spar.simulator.contract import Abort, Action, RequestUserConfirmation, ToolResponse
from spar.simulator.enums import FsmState
from spar.simulator.schemas import Sample
from spar.simulator.world import World


@dataclass
class EpisodeTrace:
    sample_id: str
    final_state: FsmState         # AGENT-terminal the loop stopped at (SETTLED/ABORTED/ESCALATED)
    grade_terminal: FsmState      # GRADE-terminal after the deferred drain (CLOSED/DISPUTED/…)
    action_log: list[Action] = field(default_factory=list)
    tool_responses: list[ToolResponse] = field(default_factory=list)  # additive (M2)
    abort_reason: str | None = None
    made_decision: bool = False   # F6: an explicit deliberate abort/escalate occurred (M2)
    completed_route_id: str | None = None  # M3: acquirer captured on a settled purchase
    hidden_final_state: dict[str, Any] = field(default_factory=dict)


def run_episode(sample: Sample, agent: Agent, *, trial_index: int = 0) -> EpisodeTrace:
    world = World(sample, trial_index=trial_index)
    obs = world.reset()
    abort_reason: str | None = None
    made_decision = False
    while not world.is_agent_terminal():
        if world.elapsed_steps >= sample.world_config.max_steps:
            # F6: a step-budget abort is NOT a deliberate decision.
            world.state = FsmState.ABORTED
            abort_reason = "step_budget_exhausted"
            break
        action = agent.act(obs)
        if isinstance(action, (Abort, RequestUserConfirmation)):
            made_decision = True
        world.step(action)
        obs = world.observe()
    # Root-cause fix (C1/C3): drain deferred events to resolve the GRADE-terminal BEFORE grading.
    # M2 drain is the identity SETTLED->CLOSED; M5's real drain can yield DISPUTED->CLOSED.
    grade_terminal = world.drain_deferred()
    return EpisodeTrace(
        sample_id=sample.sample_id,
        final_state=world.state,
        grade_terminal=grade_terminal,
        action_log=list(world.action_log),
        tool_responses=list(world._tool_responses),
        abort_reason=abort_reason,
        made_decision=made_decision,
        completed_route_id=world.completed_route_id,
        hidden_final_state=dict(world.hidden_final_state),
    )
