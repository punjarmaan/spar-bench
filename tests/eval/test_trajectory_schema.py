"""The enriched trajectory schema models + JSON Schema export (enrichment T2)."""
from spar.eval.agent import CallUsage
from spar.eval.trajectory import AgentTurn, EpisodeRecord, TRAJECTORY_SCHEMA_VERSION


def test_agent_turn_holds_full_verbatim_fields():
    t = AgentTurn(
        index=0, observation={"mandate": {"goal": "buy"}},
        reasoning="thinking hard", raw_output='{"tool":"capture","args":{}}',
        retried=False, retry_raw_output=None, retry_reasoning=None,
        action={"tool": "capture", "args": {}},
        usage=[CallUsage(prompt_tokens=10, completion_tokens=3, response_cost=0.001)],
    )
    assert t.reasoning == "thinking hard"
    assert t.usage[0].prompt_tokens == 10


def test_episode_record_round_trips_json():
    rec = EpisodeRecord(
        schema_version=TRAJECTORY_SCHEMA_VERSION, sample_id="s1", trial_index=0,
        axis="routing", intent_spec="explicit", is_trap=False,
        model="m", route="openrouter/test/m", scaffold_version="2.2.0",
        status="scored", final_state="SETTLED", grade_terminal="CLOSED",
        abort_reason=None, terminating_action="capture", system_prompt="SYS",
        turns=[], episode={"unsafe_completion": False},
    )
    dumped = rec.model_dump_json()
    again = EpisodeRecord.model_validate_json(dumped)
    assert again.sample_id == "s1"
    assert again.schema_version == TRAJECTORY_SCHEMA_VERSION


def test_json_schema_export_lists_core_fields():
    schema = EpisodeRecord.model_json_schema()
    props = schema["properties"]
    for field in ("sample_id", "trial_index", "turns", "episode", "system_prompt", "status"):
        assert field in props
