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
/ post_purchase) PASS (Plan A stamped intended_violation_class), and B2a-d re-pointed all 4
non-catastrophic axes at their real mechanisms (decline_recovery -> hard decline, stale_state ->
price drift, fraud_reactivity -> guaranteed soft block, routing -> bad route / unsupported
acquirer), so EVERY axis now has 0 offenders. The gate is still wired LOG-ONLY (enforce=False)
into `build()`; Plan B B2e flips the WIRING to enforced now that every axis is fixed.
"""

from __future__ import annotations

from spar.dataset.build import (
    COVERAGE_ENFORCED_SPLITS,
    trap_mechanism_spotcheck,
)
from spar.dataset.generator import generate
from spar.dataset.gold_backbone import diamond_backbone
from spar.dataset.plan import plan_all
from spar.simulator.enums import Axis


def _private_samples() -> list:
    """The `private` split as built: every procedural sample (full graded copy) + the
    hand-authored diamond backbone. Mirrors build()'s `private` assembly."""
    private = [generate(p.spec) for p in plan_all(build_seed=1)]
    private.extend(diamond_backbone())
    return private

_CATASTROPHIC_AXES = ("consent_mandate", "compliance_tax", "post_purchase")
_NON_CATASTROPHIC_AXES = ("routing", "decline_recovery", "stale_state", "fraud_reactivity")
# Plan B B2a fixed decline_recovery (its trap now scripts a real hard decline); B2b fixed
# stale_state (its trap now scripts a real price drift); B2c fixed fraud_reactivity (its trap now
# guarantees a soft block); B2d fixed routing (its trap is now a bad-routing / unsupported-acquirer
# surface, not the universal over-limit knob). All 4 non-catastrophic axes are now fixed.
_FIXED_AXES = ("decline_recovery", "stale_state", "fraud_reactivity", "routing")
_STILL_OFFENDING_AXES: tuple[str, ...] = ()


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


def test_fixed_axes_have_no_offenders():
    # B2a re-pointed the decline_recovery trap at a real hard decline, B2b re-pointed the
    # stale_state trap at a real price drift, and B2c re-pointed the fraud_reactivity trap at a
    # guaranteed soft block, so none of the three offends.
    counts = trap_mechanism_spotcheck(_main_samples(), enforce=False, split="main")
    for axis in _FIXED_AXES:
        assert counts.get(axis, 0) == 0, (
            f"fixed axis {axis} should have no trap-mechanism offenders, "
            f"got {counts.get(axis)}"
        )


def test_all_non_catastrophic_axes_now_fixed_issue1_closed():
    # Issue-1 is fully closed: B2a-d re-pointed every non-catastrophic axis at its real mechanism,
    # so NONE has offenders. (Was: routing still offended pending B2d.)
    counts = trap_mechanism_spotcheck(_main_samples(), enforce=False, split="main")
    offending_axes = [a for a in _NON_CATASTROPHIC_AXES if counts.get(a, 0) > 0]
    assert offending_axes == [], (
        f"expected all non-catastrophic axes fixed after B2a-d; still offending: "
        f"{offending_axes} with counts {counts}"
    )
    assert list(_STILL_OFFENDING_AXES) == [], "no axis should remain on the Issue-1 list"


def test_log_only_never_raises_with_zero_offenders():
    # enforce=False must never raise; with B2a-d landed, every axis is 0 offenders.
    counts = trap_mechanism_spotcheck(_main_samples(), enforce=False, split="main")
    assert sum(counts.values()) == 0  # Issue-1 closed: no offenders remain
    # no exception => log-only contract honored


def test_enforce_does_not_raise_when_no_offenders():
    # The FUNCTION enforces on an enforced split, but with every axis fixed there are no offenders
    # to raise on -> it returns cleanly (B2e flips the WIRING to enforced safely).
    assert "main" in COVERAGE_ENFORCED_SPLITS
    counts = trap_mechanism_spotcheck(_main_samples(), enforce=True, split="main")
    assert sum(counts.values()) == 0  # no offenders => no TrapMechanismGateError


def test_enforce_does_not_raise_off_enforced_splits():
    # A non-enforced split (lite) must not raise when enforce flows through
    # COVERAGE_ENFORCED_SPLITS membership (mirrors f1/coverage/solvability/safe_completion wiring).
    assert "lite" not in COVERAGE_ENFORCED_SPLITS
    counts = trap_mechanism_spotcheck(
        _main_samples(), enforce=("lite" in COVERAGE_ENFORCED_SPLITS), split="lite"
    )
    assert isinstance(counts, dict)  # computed, never raised


def test_enforce_true_on_private_does_not_raise_with_diamond_backbone():
    # B2e: the `private` split includes the hand-authored diamond backbone golds. Before B2e the
    # two consent backbone golds (spar_consent_mandate_post_revocation,
    # spar_consent_mandate_underspecified) surfaced as `consent_mandate: 2` offenders, so
    # enforce=True on private WOULD raise. After stamping post_revocation mandate_breach and
    # broadening the consent predicate to accept revocation/over-limit/escalation mechanisms,
    # enforce=True on the FULL built private sample set must NOT raise.
    assert "private" in COVERAGE_ENFORCED_SPLITS
    counts = trap_mechanism_spotcheck(_private_samples(), enforce=True, split="private")
    assert counts.get("consent_mandate", 0) == 0, (
        f"consent backbone golds must pass the broadened predicate, got {counts}"
    )
    assert sum(counts.values()) == 0


def test_diamond_backbone_consent_golds_pass_predicate():
    # The two hand-authored consent backbone golds must each pass the broadened consent predicate:
    # post_revocation via its (now-stamped) mandate_breach / revocation mechanism, underspecified
    # via the ESCALATED + request_user_confirmation scope-wall/escalation construct.
    counts = trap_mechanism_spotcheck(diamond_backbone(), enforce=False, split="diamond")
    assert counts.get("consent_mandate", 0) == 0, (
        f"diamond consent backbone golds should not offend, got {counts}"
    )
