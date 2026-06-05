"""EM4 — offline: the shipped example configs parse and match the documented roster/Profile A."""

from __future__ import annotations

from pathlib import Path

from spar.eval.models import load_models
from spar.eval.profile import load_profile

CONFIGS = Path(__file__).resolve().parents[2] / "configs"


def test_models_toml_parses_roster() -> None:
    models = load_models(CONFIGS / "models.toml")
    ids = {m.id for m in models}
    # Roster upgraded 2026-06-02 to current-gen flagships + curated open-weight set;
    # mistral-small-2603 + mistral-medium-3-5 added 2026-06-03: 6 frontier + 8 open-weight = 14 total.
    assert len(models) == 14
    assert sum(1 for m in models if m.cls == "frontier") == 6
    assert sum(1 for m in models if m.cls == "open") == 8

    _ALLOWED = ("openrouter/", "anthropic/", "openai/")
    assert all(any(m.route.startswith(p) for p in _ALLOWED) for m in models)
    by_id = {m.id: m.route for m in models}
    assert by_id["claude-opus-4.8"].startswith("anthropic/")
    assert by_id["claude-sonnet-4.6"].startswith("anthropic/")
    assert len(ids) == len(models)  # ids are unique + filename-safe
    assert all("/" not in m.id and " " not in m.id for m in models)


def test_profile_toml_is_profile_a() -> None:
    prof = load_profile(CONFIGS / "profile.toml")
    assert prof.competence.temperature == 0.0
    assert prof.reliability.temperature == 0.7
    plan = {(p.split, p.k, p.stage, p.published) for p in prof.plan}
    assert ("main", 1, "competence", True) in plan
    assert ("redline", 4, "reliability", True) in plan
    assert ("lite", 1, "competence", False) in plan
