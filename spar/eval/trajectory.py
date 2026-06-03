"""Enriched trajectory schema. These pydantic models are the SOURCE OF TRUTH for the on-disk
per-trial trajectory records and the exported JSON Schema / TS types in the viewer bundle.
Additive + score-neutral: nothing here is read by the grader or report."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from spar.eval.agent import CallUsage

TRAJECTORY_SCHEMA_VERSION = 1


class AgentTurn(BaseModel):
    """One agent act() turn, verbatim. `usage` has two entries when a reformat retry fired."""

    index: int
    observation: dict[str, Any]
    reasoning: str | None
    raw_output: str
    retried: bool
    retry_raw_output: str | None
    retry_reasoning: str | None
    action: dict[str, Any]
    usage: list[CallUsage]


class EpisodeTurn(AgentTurn):
    """An AgentTurn merged with the world's reaction (filled by the orchestrator)."""

    tool_response: dict[str, Any] | None = None
    user_response: dict[str, Any] | None = None


class EpisodeRecord(BaseModel):
    """One pass^k trial of one sample — the full viewer-ready record (one JSONL line)."""

    schema_version: int
    sample_id: str
    trial_index: int
    axis: str
    intent_spec: str
    is_trap: bool
    model: str
    route: str
    scaffold_version: str
    status: str
    final_state: str | None
    grade_terminal: str | None
    abort_reason: str | None
    terminating_action: str | None
    system_prompt: str
    turns: list[EpisodeTurn]
    episode: dict[str, Any]
