"""Redline is the safety-reliability split: every trap must configure its mechanism AND be
trippable by its violating agent. A defanged redline trap must fail the build."""
from __future__ import annotations

import pytest

from spar.dataset.build import redline_trippability_spotcheck, RedlineTrippabilityError
from spar.dataset.gold_backbone import redline_backbone


def test_real_redline_is_fully_trippable():
    redline_trippability_spotcheck(redline_backbone())  # must not raise


def test_defanged_redline_trap_raises():
    samples = redline_backbone()
    trap = next(s for s in samples if s.is_trap and s.gold.intended_violation_class is not None)
    defanged = trap.model_copy(update={
        "world_config": trap.world_config.model_copy(update={"issuer_behavior": {}, "decline_plan": {}, "dispute": {}})
    })
    with pytest.raises(RedlineTrippabilityError, match=trap.sample_id[:20]):
        redline_trippability_spotcheck([defanged])
