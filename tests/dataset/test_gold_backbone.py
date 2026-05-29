"""M7 Task 7b: hand-authored gold backbone -> human-verified Diamond (F14).

Reconciled to the actual M2-M5 gold id scheme (`spar_<axis>_<label>`): the F14 invariant is
that Diamond is hand-authored (never procedural). The discriminator is id-shape — a procedural
id carries a difficulty segment; hand-authored ids never do — not a synthetic `spar_gold_`
prefix, so no existing gold file is renamed.
"""

from __future__ import annotations

from spar.dataset.gold_backbone import (
    GOLD_AXES,
    diamond_backbone,
    is_hand_authored_id,
    load_gold_backbone,
)
from spar.simulator.schemas import Sample


def test_backbone_loads_hand_authored_samples():
    backbone = load_gold_backbone()
    assert backbone, "gold backbone is empty — Diamond + gold_replay gate have nothing to run against"
    assert all(isinstance(s, Sample) for s in backbone)


def test_diamond_backbone_is_all_diamond_flagged_and_hand_authored():
    diamond = diamond_backbone()
    assert diamond, "no diamond:true samples in the hand-authored backbone"
    # F14: every Diamond sample is hand-authored (diamond:true), never procedural.
    assert all(s.diamond is True for s in diamond)
    assert all(is_hand_authored_id(s.sample_id) for s in diamond)


def test_no_procedural_id_shape_in_backbone():
    # Procedural ids (GenSpec.sample_id) carry a difficulty segment; hand-authored ids never do.
    for s in load_gold_backbone():
        assert is_hand_authored_id(s.sample_id), f"{s.sample_id} looks procedural"


def test_all_seven_axes_covered_and_decline_recovery_present():
    assert len(GOLD_AXES) == 7
    assert "decline_recovery" in GOLD_AXES
    dr = [s for s in load_gold_backbone() if s.axis.value == "decline_recovery"]
    assert dr, "the M2-authored decline_recovery gold seed is missing"
