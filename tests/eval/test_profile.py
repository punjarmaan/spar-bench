import pytest

from spar.eval.profile import StagePlan, StageSampling


def test_stage_sampling_defaults():
    s = StageSampling(temperature=0.0)
    assert s.temperature == 0.0
    assert s.top_p == 1.0
    assert s.max_tokens == 2048
    assert s.seed is None


def test_stage_sampling_is_frozen():
    s = StageSampling(temperature=0.7)
    with pytest.raises(Exception):
        s.temperature = 0.0  # type: ignore[misc]


def test_stage_plan_fields():
    p = StagePlan(split="main", k=1, stage="competence", published=True)
    assert p.split == "main"
    assert p.k == 1
    assert p.stage == "competence"
    assert p.published is True
    with pytest.raises(Exception):
        StagePlan(split="x", k=1, stage="not-a-stage", published=False)  # type: ignore[arg-type]
