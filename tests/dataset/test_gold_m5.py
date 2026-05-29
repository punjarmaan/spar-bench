"""Structural validity of the four M5 axis gold files (one legit + one trap each)."""

from __future__ import annotations

from importlib import resources

from spar.simulator.enums import Axis, FsmState
from spar.simulator.schemas import Sample

_AXES = ["stale_state", "compliance_tax", "fraud_reactivity", "post_purchase"]


def _load(axis: str) -> list[Sample]:
    text = resources.files("spar.dataset.gold").joinpath(f"{axis}.jsonl").read_text("utf-8")
    return [Sample.model_validate_json(line) for line in text.splitlines() if line.strip()]


def test_every_axis_has_at_least_one_legit_and_one_trap():
    for axis in _AXES:
        samples = _load(axis)
        assert any(not s.is_trap for s in samples), f"{axis} has no legit sample"
        assert any(s.is_trap for s in samples), f"{axis} has no trap sample"


def test_axis_field_matches_filename():
    for axis in _AXES:
        for s in _load(axis):
            assert s.axis is Axis(axis)


def test_stale_state_trap_requires_revalidate_and_forbids_stale_pay():
    traps = [s for s in _load("stale_state") if s.is_trap]
    t = traps[0]
    assert "revalidate_before_submit" in t.gold.must
    assert "pay_at_stale_price" in t.gold.must_not
    assert t.gold.correct_outcome in {FsmState.ABORTED, FsmState.ESCALATED}


def test_compliance_tax_legit_requires_compute_tax():
    legit = [s for s in _load("compliance_tax") if not s.is_trap][0]
    assert "compute_tax" in legit.gold.must


def test_post_purchase_legit_requires_verify_capture_result():
    legit = [s for s in _load("post_purchase") if not s.is_trap][0]
    assert "verify_capture_result" in legit.gold.must
