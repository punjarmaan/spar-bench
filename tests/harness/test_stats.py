"""Wilson score interval properties (C11)."""

from __future__ import annotations

from spar.harness.stats import wilson_interval


def test_n_zero_is_full_interval():
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_all_failures_low_is_zero_high_bounded():
    low, high = wilson_interval(0, 20)
    assert low == 0.0
    assert 0.0 < high < 1.0


def test_all_successes_high_is_one_low_bounded():
    low, high = wilson_interval(20, 20)
    assert high == 1.0
    assert 0.0 < low < 1.0


def test_half_brackets_point_five():
    low, high = wilson_interval(5, 10)
    assert low < 0.5 < high


def test_within_unit_and_ordered():
    for s, n in [(0, 0), (0, 20), (20, 20), (5, 10), (3, 7), (1, 100)]:
        low, high = wilson_interval(s, n)
        assert 0.0 <= low <= 1.0
        assert 0.0 <= high <= 1.0
        assert low <= high
