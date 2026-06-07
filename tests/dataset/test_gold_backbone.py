"""Hand-authored gold backbone -> human-verified Redline.

Reconciled to the actual gold id scheme (`spar_<axis>_<label>`): the invariant is
that Redline is hand-authored (never procedural). The discriminator is id-shape — a procedural
id carries a difficulty segment; hand-authored ids never do — not a synthetic `spar_gold_`
prefix, so no existing gold file is renamed.
"""

from __future__ import annotations

from spar.dataset.gold_backbone import (
    GOLD_AXES,
    redline_backbone,
    is_hand_authored_id,
    load_gold_backbone,
)
from spar.simulator.schemas import Sample


def test_backbone_loads_hand_authored_samples():
    backbone = load_gold_backbone()
    assert backbone, "gold backbone is empty — Redline + gold_replay gate have nothing to run against"
    assert all(isinstance(s, Sample) for s in backbone)


def test_redline_backbone_is_all_redline_flagged_and_hand_authored():
    redline = redline_backbone()
    assert redline, "no redline:true samples in the hand-authored backbone"
    # every Redline sample is hand-authored (redline:true), never procedural.
    assert all(s.redline is True for s in redline)
    assert all(is_hand_authored_id(s.sample_id) for s in redline)


def test_no_procedural_id_shape_in_backbone():
    # Procedural ids (GenSpec.sample_id) carry a difficulty segment; hand-authored ids never do.
    for s in load_gold_backbone():
        assert is_hand_authored_id(s.sample_id), f"{s.sample_id} looks procedural"


def test_all_seven_axes_covered_and_decline_recovery_present():
    assert len(GOLD_AXES) == 7
    assert "decline_recovery" in GOLD_AXES
    dr = [s for s in load_gold_backbone() if s.axis.value == "decline_recovery"]
    assert dr, "the hand-authored decline_recovery gold seed is missing"
