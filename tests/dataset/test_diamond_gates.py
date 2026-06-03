"""Diamond is the safety-reliability split: every trap must configure its mechanism AND be
trippable by its violating agent. A defanged diamond trap must fail the build."""
from __future__ import annotations

import pytest

from spar.dataset.build import diamond_trippability_spotcheck, DiamondTrippabilityError
from spar.dataset.gold_backbone import diamond_backbone


def test_real_diamond_is_fully_trippable():
    diamond_trippability_spotcheck(diamond_backbone())  # must not raise


def test_defanged_diamond_trap_raises():
    samples = diamond_backbone()
    trap = next(s for s in samples if s.is_trap and s.gold.intended_violation_class is not None)
    defanged = trap.model_copy(update={
        "world_config": trap.world_config.model_copy(update={"issuer_behavior": {}, "decline_plan": {}, "dispute": {}})
    })
    with pytest.raises(DiamondTrippabilityError, match=trap.sample_id[:20]):
        diamond_trippability_spotcheck([defanged])
