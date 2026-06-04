from __future__ import annotations

from collections import Counter

from spar.dataset.gold_backbone import redline_backbone
from spar.simulator.enums import Axis


def test_each_axis_has_at_least_two_trap_anchors():
    redline = redline_backbone()
    per_axis_traps = Counter(s.axis for s in redline if s.is_trap)
    for a in Axis:
        assert per_axis_traps[a.value] >= 2, (
            f"{a.value} has {per_axis_traps[a.value]} trap anchors (<2)"
        )


def test_anchors_span_three_catastrophic_classes():
    classes = {
        s.gold.intended_violation_class
        for s in redline_backbone()
        if s.is_trap and s.gold.intended_violation_class is not None
    }
    assert len(classes) >= 3, f"only {len(classes)} catastrophic classes in anchors"


def test_every_anchor_trap_passes_redline_gates_and_is_safe_under_diligent():
    from spar.dataset.build import (
        redline_trippability_spotcheck,
        trap_mechanism_spotcheck,
    )
    from spar.agents.reference_agents import DiligentAgent
    from spar.harness.runner import run_episode
    from spar.harness.graders import score

    anchors = [s for s in redline_backbone() if s.is_trap]
    trap_mechanism_spotcheck(
        anchors, enforce=True, split="redline", enforced_splits=frozenset({"redline"})
    )
    redline_trippability_spotcheck(anchors)
    for s in anchors:
        sc = score(s, run_episode(s, DiligentAgent(), trial_index=0))
        assert sc.catastrophic_class is None and not sc.unsafe_completion, (
            f"{s.sample_id} unsafe under DiligentAgent"
        )
