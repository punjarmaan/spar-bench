from importlib import resources

from spar.agents.reference_agents import AlwaysCompleteAgent, GoldReplayAgent
from spar.harness.graders import score
from spar.harness.runner import run_episode
from spar.simulator.enums import Axis, FsmState
from spar.simulator.schemas import Sample


def _load() -> list[Sample]:
    text = (
        resources.files("spar.dataset.gold")
        .joinpath("decline_recovery.jsonl")
        .read_text(encoding="utf-8")
    )
    return [Sample.model_validate_json(line) for line in text.splitlines() if line.strip()]


def test_ten_gold_samples_all_decline_recovery_scripted():
    samples = _load()
    assert len(samples) == 10
    for s in samples:
        assert s.axis is Axis.DECLINE_RECOVERY
        assert s.world_config.decline_plan.get("mode") == "scripted"
        # Root-cause fix: completing gold is CLOSED (never SETTLED); abort traps are ABORTED.
        assert s.gold.correct_outcome in (FsmState.CLOSED, FsmState.ABORTED)
        assert s.gold.trajectory is not None  # every gold carries a verbatim replay trajectory


def test_no_gold_uses_settled_as_correct_outcome():
    # Guards the root-cause fix (C1/C3): SETTLED-as-gold would defeat the M5 retroactive dispute.
    for s in _load():
        assert s.gold.correct_outcome is not FsmState.SETTLED


def test_set_spans_required_code_families():
    samples = _load()
    codes = {
        d["reason"]
        for s in samples
        for d in s.world_config.decline_plan["draws"]
        if d.get("outcome") == "decline"
    }
    # soft, ambiguous, hard, 14, 54 all present.
    assert {"51", "91", "65"} & codes  # soft family
    assert "05" in codes               # ambiguous
    assert {"43", "46", "62"} & codes  # hard family
    assert "14" in codes and "54" in codes
    # at least one 3DS challenge sample.
    assert any(
        any(d.get("outcome") == "challenge" for d in s.world_config.decline_plan["draws"])
        for s in samples
    )
    # the Redline soft-then-hard flip.
    assert any(s.redline for s in samples)


def test_gold_replay_agent_scores_near_one_across_the_set():
    samples = _load()
    scores = [
        score(s, run_episode(s, GoldReplayAgent(trajectory=s.gold.trajectory), trial_index=0))
        for s in samples
    ]
    mean = sum(x.score for x in scores) / len(scores)
    assert mean >= 0.95
    assert all(x.outcome_correct for x in scores)


def test_always_complete_agent_scores_low_on_hard_decline_traps():
    samples = [s for s in _load() if s.is_trap]
    assert samples  # there are hard-decline traps
    scores = [
        score(s, run_episode(s, AlwaysCompleteAgent(), trial_index=0)) for s in samples
    ]
    assert all(x.score <= 0.0 for x in scores)
    assert all(not x.outcome_correct for x in scores)
