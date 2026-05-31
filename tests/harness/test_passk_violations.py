"""C11: lock the catastrophic-trial-is-never-solved invariant.

A catastrophic-applicable sample is zeroed by Task 3.2 (partitioned score = 0.0).
is_solved(0.0) must be False under both binary and routing thresholds so that
run_trials never counts a violated trial as solved.
"""

from spar.harness.passk import is_solved


def test_catastrophic_trial_is_never_solved():
    # A violated trial scores 0.0 -> below both thresholds -> unsolved.
    assert is_solved(0.0, binary=True) is False
    assert is_solved(0.0, binary=False) is False
