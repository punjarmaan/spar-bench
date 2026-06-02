"""EM4 — offline: the shipped example configs parse and match the documented roster/Profile A."""

from __future__ import annotations

from pathlib import Path

from spar.eval.models import load_models
from spar.eval.profile import load_profile

CONFIGS = Path(__file__).resolve().parents[2] / "configs"


def test_models_toml_parses_roster() -> None:
    models = load_models(CONFIGS / "models.toml")
    ids = {m.id for m in models}
    # Roster grew to 10: 5 frontier + 5 open (3 free :free probes added in probe/free-model pass).
    assert len(models) == 10
    assert sum(1 for m in models if m.cls == "frontier") == 5
    assert sum(1 for m in models if m.cls == "open") == 5
    assert all(m.route.startswith("openrouter/") for m in models)
    assert len(ids) == len(models)  # ids are unique + filename-safe
    assert all("/" not in m.id and " " not in m.id for m in models)


def test_profile_toml_is_profile_a() -> None:
    prof = load_profile(CONFIGS / "profile.toml")
    assert prof.competence.temperature == 0.0
    assert prof.reliability.temperature == 0.7
    plan = {(p.split, p.k, p.stage, p.published) for p in prof.plan}
    assert ("main", 1, "competence", True) in plan
    assert ("diamond", 4, "reliability", True) in plan
    assert ("lite", 1, "competence", False) in plan
