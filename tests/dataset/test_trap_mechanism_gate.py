"""Plan B B1c (STATIC, LOG-ONLY for now): build-time trap-mechanism gate.

Each axis's TRAP samples are supposed to CONFIGURE the world mechanism that the trap's
`must_not` token references, so the token can actually fire under an over-completing agent.
Issue-1 defect: the current procedural generator routes routing/decline_recovery/stale_state/
fraud_reactivity traps through the UNIVERSAL over-limit trap (`per_txn_max = cart_total - 1`)
and merely OVERRIDES `gold.must_not` — without configuring any decline/drift/fraud/route
mechanic. So those traps trip on `exceed_per_txn_max` scope, NEVER on their axis token.

This gate is a STATIC, no-agent-run check on the sample: per-axis, for each TRAP, it asserts
the mechanism predicate holds (an unsupported route exists / a HARD decline draw exists / a
price-drift knob is set / the fraud engine is parameterized to guarantee a soft block / the
catastrophic gold carries an intended_violation_class). It is wired LOG-ONLY (enforce=False)
into `build()` now; Plan B B2e flips it to enforced once B2a-d fix the four broken axes.

These tests DOCUMENT the current state: the 3 catastrophic axes (consent_mandate / compliance_tax
/ post_purchase) already PASS (Plan A stamped intended_violation_class), while the 4
non-catastrophic axes currently have offenders (Issue-1). We assert the catastrophic axes pass
and RECORD the others; we deliberately do NOT assert all axes pass — they don't yet.
"""

from __future__ import annotations

import pytest

from spar.dataset.build import (
    COVERAGE_ENFORCED_SPLITS,
    TrapMechanismGateError,
    trap_mechanism_spotcheck,
)
from spar.dataset.generator import generate
from spar.dataset.plan import plan_all
from spar.simulator.enums import Axis

_CATASTROPHIC_AXES = ("consent_mandate", "compliance_tax", "post_purchase")
_NON_CATASTROPHIC_AXES = ("routing", "decline_recovery", "stale_state", "fraud_reactivity")


def _main_samples() -> list:
    return [generate(p.spec) for p in plan_all(build_seed=1) if p.split == "main"]


def test_trap_mechanism_spotcheck_is_exported_and_callable():
    assert callable(trap_mechanism_spotcheck)


def test_returns_per_axis_offender_counts_dict():
    counts = trap_mechanism_spotcheck(_main_samples(), enforce=False, split="main")
    assert isinstance(counts, dict)
    # Every axis with >=1 trap appears in the dict (zero or more offenders).
    assert set(counts) <= {a.value for a in Axis}
    assert all(isinstance(v, int) and v >= 0 for v in counts.values())


def test_catastrophic_axes_have_no_offenders():
    # Plan A stamped intended_violation_class on every catastrophic trap, so all 3 PASS today.
    counts = trap_mechanism_spotcheck(_main_samples(), enforce=False, split="main")
    for axis in _CATASTROPHIC_AXES:
        assert counts.get(axis, 0) == 0, (
            f"catastrophic axis {axis} should have no trap-mechanism offenders, "
            f"got {counts.get(axis)}"
        )


def test_non_catastrophic_axes_currently_have_offenders_issue1():
    # DOCUMENTS the Issue-1 gap: the 4 non-catastrophic axes' traps are the universal over-limit
    # trap and DO NOT configure their axis mechanism, so every one is an offender today. This is
    # expected to flip to zero after Plan B B2a-d. Recorded, not asserted away.
    counts = trap_mechanism_spotcheck(_main_samples(), enforce=False, split="main")
    offending_axes = [a for a in _NON_CATASTROPHIC_AXES if counts.get(a, 0) > 0]
    assert offending_axes == list(_NON_CATASTROPHIC_AXES), (
        "expected ALL four non-catastrophic axes to have offenders on the current generator "
        f"(Issue-1); got offending axes {offending_axes} with counts {counts}"
    )


def test_log_only_never_raises_even_with_offenders():
    # enforce=False must never raise, regardless of offenders (the wiring is log-only now).
    counts = trap_mechanism_spotcheck(_main_samples(), enforce=False, split="main")
    assert sum(counts.values()) > 0  # Issue-1: offenders exist
    # no exception => log-only contract honored


def test_enforce_raises_on_enforced_split_when_offenders_exist():
    # The FUNCTION can enforce: on an enforced split with offenders it raises (B2e will flip the
    # WIRING to this). Today main has offenders (Issue-1), so enforcing must raise.
    assert "main" in COVERAGE_ENFORCED_SPLITS
    with pytest.raises(TrapMechanismGateError):
        trap_mechanism_spotcheck(_main_samples(), enforce=True, split="main")


def test_enforce_does_not_raise_off_enforced_splits():
    # Even with offenders, a non-enforced split (lite) must not raise when enforce flows through
    # COVERAGE_ENFORCED_SPLITS membership (mirrors f1/coverage/solvability/safe_completion wiring).
    assert "lite" not in COVERAGE_ENFORCED_SPLITS
    counts = trap_mechanism_spotcheck(
        _main_samples(), enforce=("lite" in COVERAGE_ENFORCED_SPLITS), split="lite"
    )
    assert isinstance(counts, dict)  # computed, never raised
