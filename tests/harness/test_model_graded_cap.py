"""The Tier-C model-graded weight fraction is a hard <10% gate."""

from __future__ import annotations

import pytest

from spar.harness.graders import SampleScore
from spar.harness.report import ModelGradedCapExceeded, build_results
from spar.simulator.enums import FsmState


def _scores(n_total: int, n_model_graded: int) -> list[SampleScore]:
    out = []
    for i in range(n_total):
        out.append(SampleScore(
            sample_id=f"s{i}", axis="consent_mandate", is_trap=False, score=1.0,
            outcome_correct=True, final_state=FsmState.CLOSED,
            model_graded=(i < n_model_graded),
            grader_model="stub-model-grader@1" if i < n_model_graded else None,
        ))
    return out


def test_cap_ok_under_10_percent():
    results = build_results(_scores(20, 1), split="lite", canary="spar:t", build_seed=1,
                            weights={"score_floor": -1.0})
    assert results["summary"]["model_graded_weight_fraction"] == pytest.approx(0.05)


def test_cap_exceeded_raises():
    with pytest.raises(ModelGradedCapExceeded):
        build_results(_scores(20, 3), split="lite", canary="spar:t", build_seed=1,
                      weights={"score_floor": -1.0})
