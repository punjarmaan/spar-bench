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


def test_profile_holds_two_stages_and_a_plan():
    from spar.eval.profile import Profile

    prof = Profile(
        competence=StageSampling(temperature=0.0),
        reliability=StageSampling(temperature=0.7),
        plan=[StagePlan(split="main", k=1, stage="competence", published=True)],
    )
    assert prof.competence.temperature == 0.0
    assert prof.reliability.temperature == 0.7
    assert len(prof.plan) == 1
    with pytest.raises(Exception):
        prof.plan = []  # type: ignore[misc]


def test_default_profile_is_profile_a():
    from spar.eval.profile import DEFAULT_PROFILE

    # Sampling per spec §5.3: competence temp 0.0, reliability temp 0.7.
    assert DEFAULT_PROFILE.competence.temperature == 0.0
    assert DEFAULT_PROFILE.competence.top_p == 1.0
    assert DEFAULT_PROFILE.competence.max_tokens == 2048
    assert DEFAULT_PROFILE.competence.seed == 7
    assert DEFAULT_PROFILE.reliability.temperature == 0.7
    # Plan per spec §6 / index: main/k1/competence/published, redline/k4/reliability/published,
    # lite/k1/competence/unpublished.
    plan = {(p.split, p.k, p.stage, p.published) for p in DEFAULT_PROFILE.plan}
    assert plan == {
        ("main", 1, "competence", True),
        ("redline", 4, "reliability", True),
        ("lite", 1, "competence", False),
    }


def test_load_profile_reads_stages_and_plan(tmp_path):
    from spar.eval.profile import load_profile

    toml = (
        '[profile.competence]\n'
        'temperature = 0.0\n'
        'top_p = 1.0\n'
        'max_tokens = 2048\n'
        'seed = 7\n'
        '\n'
        '[profile.reliability]\n'
        'temperature = 0.7\n'
        'top_p = 1.0\n'
        'max_tokens = 2048\n'
        '\n'
        '[[plan]]\n'
        'split = "main"\n'
        'k = 1\n'
        'stage = "competence"\n'
        'published = true\n'
        '\n'
        '[[plan]]\n'
        'split = "redline"\n'
        'k = 4\n'
        'stage = "reliability"\n'
        'published = true\n'
    )
    path = tmp_path / "profile.toml"
    path.write_text(toml, encoding="utf-8")

    prof = load_profile(path)
    assert prof.competence.temperature == 0.0
    assert prof.competence.seed == 7
    assert prof.reliability.temperature == 0.7
    assert prof.reliability.seed is None
    assert [(p.split, p.k, p.stage) for p in prof.plan] == [
        ("main", 1, "competence"),
        ("redline", 4, "reliability"),
    ]
