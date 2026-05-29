from collections import Counter

from spar.simulator.enums import Axis
from spar.dataset.plan import (
    plan_all, plan_samples, PROCEDURAL_SPLITS, TARGET_TRAP_FRACTION,
)


def test_plan_is_deterministic_in_seed():
    a = plan_all(build_seed=1)
    b = plan_all(build_seed=1)
    assert [(p.split, p.spec.sample_id) for p in a] == \
        [(p.split, p.spec.sample_id) for p in b]


def test_split_membership_is_a_pure_function_of_sample_id_and_seed():
    # The single mechanism: a sample's split is fixed by the enumeration, with no
    # second hash pass. Every sample_id maps to exactly one split, stably.
    plan = plan_all(build_seed=1)
    by_id = {p.spec.sample_id: p.split for p in plan}
    assert len(by_id) == len(plan)  # no sample_id appears in two splits
    again = {p.spec.sample_id: p.split for p in plan_all(build_seed=1)}
    assert by_id == again


def test_plan_covers_only_procedural_public_splits():
    # Diamond is the hand-authored backbone (F14), never procedural here.
    splits = {p.split for p in plan_all(build_seed=1)}
    assert splits == set(PROCEDURAL_SPLITS) == {"lite", "main"}


def test_lite_is_balanced_across_all_seven_axes():
    specs = plan_samples("lite", build_seed=1)
    by_axis = Counter(s.axis for s in specs)
    assert set(by_axis) == set(Axis)
    counts = list(by_axis.values())
    assert max(counts) - min(counts) <= 2  # balanced


def test_trap_fraction_near_target_per_split_per_axis():
    # Per-split-per-axis (not just Main): the plan establishes balance everywhere.
    for split in PROCEDURAL_SPLITS:
        specs = plan_samples(split, build_seed=1)
        for axis in Axis:
            axis_specs = [s for s in specs if s.axis is axis]
            frac = sum(1 for s in axis_specs if s.is_trap) / len(axis_specs)
            assert abs(frac - TARGET_TRAP_FRACTION) <= 0.05  # within 5pp


def test_model_graded_eligible_fraction_under_cap():
    from spar.dataset.generator import generate
    specs = plan_samples("main", build_seed=1)
    samples = [generate(s) for s in specs]
    mg = sum(1 for s in samples if s.model_graded) / len(samples)
    assert mg < 0.10  # F15 / module 40 §3.3 cap


def test_lite_size_in_expected_band():
    specs = plan_samples("lite", build_seed=1)
    assert 42 <= len(specs) <= 84  # ~60, balanced 7 axes
