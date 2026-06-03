"""Task 4.3b (C8): build-time per-class catastrophic coverage gate + LateRefundAgent."""

from __future__ import annotations

import pytest

from spar.agents.reference_agents import LateRefundAgent
from spar.dataset.build import (
    MIN_APPLICABLE_N,
    CoverageGateError,
    coverage_spotcheck,
)
from spar.dataset.applicability import applicability_from_world_config
from spar.dataset.generator import generate
from spar.dataset.plan import plan_samples
from spar.harness.runner import run_episode
from spar.harness.violations import FLAG_BY_CLASS
from spar.simulator.enums import Axis, Difficulty, IntentSpec, ViolationClass
from spar.dataset.generator import GenSpec


def _main_samples(seed: int = 12345):
    return [generate(s) for s in plan_samples("main", build_seed=seed)]


def _missed_reversal_trap():
    """A POST_PURCHASE trap that authors missed_reversal_window (seed 2 -> that class)."""
    return generate(
        GenSpec(axis=Axis.POST_PURCHASE, seed=2, difficulty=Difficulty.MEDIUM, is_trap=True)
    )


def test_late_refund_agent_trips_missed_reversal_window():
    sample = _missed_reversal_trap()
    assert ViolationClass.MISSED_REVERSAL_WINDOW in applicability_from_world_config(sample)
    trace = run_episode(sample, LateRefundAgent(), trial_index=0)
    flag = FLAG_BY_CLASS[ViolationClass.MISSED_REVERSAL_WINDOW]
    assert trace.hidden_final_state.get(flag) is True


def test_gate_passes_on_real_main_split():
    """Main has >=8 applicable + trippable for all 7 classes -> enforce=True must not raise."""
    report = coverage_spotcheck(_main_samples(), split="main", enforce=True)
    assert report["classes_with_coverage"] == "7/7"
    for vc in ViolationClass:
        assert report[vc.value] >= MIN_APPLICABLE_N


def test_gate_reports_classes_with_coverage_without_enforcing():
    """A tiny under-covered split is computed + reported (never silently truncated)."""
    samples = [_missed_reversal_trap()]  # one trap -> only some classes applicable, all < floor
    report = coverage_spotcheck(samples, split="lite", enforce=False)
    assert isinstance(report["classes_with_coverage"], str)
    assert report["classes_with_coverage"].endswith(f"/{len(ViolationClass)}")
    # Reported but NOT raised because enforce=False.


def test_gate_raises_when_a_class_is_under_covered():
    """A split missing the floor for a class MUST raise CoverageGateError when enforced."""
    samples = [_missed_reversal_trap()]  # nowhere near >=8 for any class
    with pytest.raises(CoverageGateError) as exc:
        coverage_spotcheck(samples, split="main", enforce=True)
    msg = str(exc.value)
    assert "main" in msg
    assert "floor" in msg
    # The specific under-covered class + its n_applicable are named (fail loud, specific).
    assert "n_applicable=" in msg


def test_gate_raises_naming_a_specific_below_floor_class():
    # Build a list applicable to overspend only a handful of times (< MIN_APPLICABLE_N).
    specs = [
        GenSpec(axis=Axis.CONSENT_MANDATE, seed=s, difficulty=Difficulty.EASY,
                is_trap=True, intent_spec=IntentSpec.EXPLICIT)
        for s in range(MIN_APPLICABLE_N - 1)
    ]
    samples = [generate(s) for s in specs]
    with pytest.raises(CoverageGateError) as exc:
        coverage_spotcheck(samples, split="main", enforce=True)
    assert "overspend" in str(exc.value)


def test_main_split_meets_floor_for_every_class():
    report = coverage_spotcheck(_main_samples(), split="main", enforce=False)
    for vc in ViolationClass:
        assert report[vc.value] >= MIN_APPLICABLE_N, f"{vc.value} below floor: {report[vc.value]}"


def test_coverage_gate_is_intended_keyed_and_matches_report_attribution():
    """Plan B B5: the build coverage gate counts a sample toward class `vc` iff `vc` is its
    gold.intended_violation_class — the SAME attribution report._per_class uses (which keys on
    SampleScore.applicable_classes == [intended]). So `n_applicable[vc]` must equal the number of
    main samples whose intended_violation_class is `vc`, NOT the broad world-config net."""
    samples = _main_samples()
    report = coverage_spotcheck(samples, split="main", enforce=False)
    expected: dict[str, int] = {vc.value: 0 for vc in ViolationClass}
    for s in samples:
        ivc = s.gold.intended_violation_class
        if ivc is not None:
            expected[ivc.value] += 1
    for vc in ViolationClass:
        assert report[vc.value] == expected[vc.value], (
            f"{vc.value}: gate {report[vc.value]} != intended count {expected[vc.value]}"
        )


def test_intended_class_distribution_clears_floor_across_build_seeds():
    """Plan B B5: round-robin catastrophic-class assignment (GenSpec.trap_index) makes the
    per-class INTENDED count seed-INVARIANT and >= MIN_APPLICABLE_N for EVERY class on main, so a
    build does not silently drop a class below the (now intended-keyed) coverage floor depending
    on the build seed (the old `seed % k` residue dipped overspend/premature_capture to 7)."""
    for seed in (1, 12345, 7, 99, 2024):
        samples = _main_samples(seed=seed)
        report = coverage_spotcheck(samples, split="main", enforce=True)
        assert report["classes_with_coverage"] == "7/7"
        for vc in ViolationClass:
            assert report[vc.value] >= MIN_APPLICABLE_N, (
                f"seed={seed} {vc.value} below floor: {report[vc.value]}"
            )
