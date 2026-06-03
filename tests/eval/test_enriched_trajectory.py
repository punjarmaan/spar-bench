"""Enriched trajectories: full schema from a replayed trace (enrichment T5)."""
from spar.eval.trajectory import EpisodeRecord
from spar.harness.runner import run_episode
from spar.eval.agent import ModelAgent
from spar.eval.profile import StageSampling
from spar.eval.orchestrator import _episode_record
from spar.dataset.loader import load_split


def _fn(content):
    def _f(**kwargs):
        msg = type("M", (), {"content": content, "reasoning_content": "rsn"})()
        choice = type("C", (), {"message": msg})()
        return type("R", (), {
            "choices": [choice],
            "usage": type("U", (), {"prompt_tokens": 7, "completion_tokens": 2})(),
            "_hidden_params": {"response_cost": 0.0},
        })()
    return _f


def test_episode_record_builder_emits_valid_schema():
    sample = load_split("lite")[0]
    agent = ModelAgent(
        route="openrouter/test/m", policy_text="p", sampling=StageSampling(temperature=0.0),
        supports_response_format=False, completion_fn=_fn('{"tool":"abort","args":{"reason":"x"}}'),
        mandate_text="m",
    )
    trace = run_episode(sample, agent, trial_index=2)
    rec = _episode_record(
        sample=sample, trace=trace, status_value="refused",
        model_id="test-model", route="openrouter/test/m", trial_index=2, system_prompt="SYS",
    )
    parsed = EpisodeRecord.model_validate(rec)
    assert parsed.sample_id == sample.sample_id
    assert parsed.trial_index == 2
    assert parsed.scaffold_version == "2.2.0"
    assert parsed.system_prompt == "SYS"
    assert parsed.turns[0].reasoning == "rsn"
    assert "tool_response" in parsed.turns[0].model_dump()
