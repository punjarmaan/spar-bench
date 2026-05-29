"""M7 Task 11: behavioral F1 spot-check — NaiveCompleteAgent fails a floor of Main non-traps."""

from __future__ import annotations

from spar.dataset.build import f1_spotcheck
from spar.dataset.generator import generate
from spar.dataset.plan import plan_samples


def test_naive_complete_fails_floor_fraction_of_main_non_traps():
    specs = [s for s in plan_samples("main", build_seed=1) if not s.is_trap]
    samples = [generate(s) for s in specs]
    failed_fraction = f1_spotcheck(samples, floor=0.9)
    assert failed_fraction >= 0.9


def test_f1_spotcheck_returns_a_fraction():
    samples = [generate(s) for s in plan_samples("lite", build_seed=1) if not s.is_trap]
    frac = f1_spotcheck(samples, floor=0.0)
    assert 0.0 <= frac <= 1.0
