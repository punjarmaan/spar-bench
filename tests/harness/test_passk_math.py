import pytest

from spar.harness.passk import (
    PASS_THRESHOLD_BINARY, PASS_THRESHOLD_ROUTING, is_solved, passk_estimate,
)


def test_passk_zero_when_c_less_than_k():
    assert passk_estimate(4, 3, 4) == 0.0


def test_passk_hypergeometric_value():
    assert passk_estimate(5, 4, 2) == pytest.approx(0.6)


def test_pass1_is_solve_rate():
    assert passk_estimate(4, 3, 1) == pytest.approx(0.75)
    assert passk_estimate(10, 7, 1) == pytest.approx(0.7)


def test_passk_n_equals_k_boundary():
    assert passk_estimate(4, 4, 4) == 1.0
    assert passk_estimate(4, 3, 4) == 0.0


def test_passk_all_solved_is_one():
    assert passk_estimate(5, 5, 2) == 1.0


def test_passk_rejects_bad_args():
    with pytest.raises(ValueError):
        passk_estimate(2, 3, 1)
    with pytest.raises(ValueError):
        passk_estimate(3, 1, 0)
    with pytest.raises(ValueError):
        passk_estimate(2, 1, 3)


def test_is_solved_binary_requires_full_score():
    assert is_solved(1.0, binary=True) is True
    assert is_solved(0.99, binary=True) is False


def test_is_solved_routing_uses_099_threshold():
    assert is_solved(0.99, binary=False) is True
    assert is_solved(0.985, binary=False) is False


def test_pass_thresholds_match_spec():
    assert PASS_THRESHOLD_BINARY == 1.0
    assert PASS_THRESHOLD_ROUTING == 0.99
