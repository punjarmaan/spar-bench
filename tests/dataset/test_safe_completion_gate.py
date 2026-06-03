"""Plan B B1b (STATIC): build-time safe-completion gate.

Every CLOSED-gold NON-TRAP is meant to be a genuinely safe completion target — a competent
single-pass completion (select -> submit -> capture -> CLOSED) must NOT be able to latch a
catastrophic flag. That can only happen if the sample's `world_config` configures a
catastrophic-divergence mechanic (a settled currency/payee that mismatches the mandate, an
overspending cart/drift, auth_only, a refund-window/dispute reversal trap, or a scope breach
that fires at the gate). This gate is a STATIC check on the sample (no agent run): it inspects
`world_config` against `mandate`/`gold` and raises if any CLOSED-gold non-trap configures a
divergence. It protects future diversity changes (new currencies/MCCs/payees) from accidentally
making a "complete me" non-trap settle in a mismatched currency / wrong payee.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from spar.dataset.build import (
    COVERAGE_ENFORCED_SPLITS,
    SafeCompletionGateError,
    safe_completion_spotcheck,
)
from spar.dataset.generator import generate
from spar.dataset.plan import plan_all
from spar.simulator.enums import FsmState


def _main_closed_non_traps(limit: int = 12):
    planned = [p for p in plan_all(build_seed=1) if p.split == "main" and not p.spec.is_trap]
    samples = [generate(p.spec) for p in planned[:limit]]
    return [s for s in samples if s.gold.correct_outcome is FsmState.CLOSED]


def test_safe_completion_spotcheck_is_exported_and_callable():
    assert callable(safe_completion_spotcheck)


def test_clean_closed_non_traps_pass_the_gate():
    # The current generator's CLOSED-gold non-traps configure NO divergence mechanic, so the
    # gate must pass (return an empty offender list) even when enforced on `main`.
    samples = _main_closed_non_traps()
    assert samples, "expected some CLOSED-gold main non-traps to sample"
    offenders = safe_completion_spotcheck(samples, enforce=True, split="main")
    assert offenders == []


def test_wrong_currency_divergence_is_caught_when_enforced():
    # Inject a settled_currency that mismatches the mandate currency on a CLOSED-gold non-trap:
    # a competent completion would latch `wrong_currency` at capture. The gate must raise.
    samples = _main_closed_non_traps()
    base = samples[0]
    bad_currency = "EUR" if base.mandate.currency != "EUR" else "GBP"
    diverged = base.model_copy(
        update={
            "world_config": base.world_config.model_copy(
                update={"issuer_behavior": {"settled_currency": bad_currency}}
            )
        }
    )
    with pytest.raises(SafeCompletionGateError) as ei:
        safe_completion_spotcheck([diverged], enforce=True, split="main")
    assert diverged.sample_id in str(ei.value)
    assert "wrong_currency" in str(ei.value)


def test_misdirected_funds_divergence_is_caught_when_enforced():
    # A settled_payee outside the mandate's merchant_constraint would latch `misdirected_funds`.
    samples = _main_closed_non_traps()
    base = samples[0]
    bad_payee = "definitely_not_an_authorized_payee_xyz"
    diverged = base.model_copy(
        update={
            "world_config": base.world_config.model_copy(
                update={"issuer_behavior": {"settled_payee": bad_payee}}
            )
        }
    )
    with pytest.raises(SafeCompletionGateError) as ei:
        safe_completion_spotcheck([diverged], enforce=True, split="main")
    assert "misdirected_funds" in str(ei.value)


def test_overspend_cart_total_divergence_is_caught_when_enforced():
    # A cart_total above the mandate amount_limit would latch `exceeded_ceiling` at approve.
    samples = _main_closed_non_traps()
    base = next((s for s in samples if s.mandate.amount_limit is not None), None)
    if base is None:
        pytest.skip("no CLOSED non-trap with an amount_limit to overspend")
    over = (base.mandate.amount_limit or Decimal("0")) + Decimal("100")
    diverged = base.model_copy(
        update={"world_config": base.world_config.model_copy(update={"cart_total": over})}
    )
    with pytest.raises(SafeCompletionGateError) as ei:
        safe_completion_spotcheck([diverged], enforce=True, split="main")
    assert "exceeded_ceiling" in str(ei.value)


def test_auth_only_divergence_is_caught_when_enforced():
    # auth_only set means a capture is a premature_capture (correct behavior is void/escalate).
    samples = _main_closed_non_traps()
    base = samples[0]
    diverged = base.model_copy(
        update={
            "world_config": base.world_config.model_copy(
                update={"issuer_behavior": {"auth_only": True}}
            )
        }
    )
    with pytest.raises(SafeCompletionGateError) as ei:
        safe_completion_spotcheck([diverged], enforce=True, split="main")
    assert "premature_capture" in str(ei.value)


def test_gate_is_log_only_when_not_enforced():
    # enforce=False must never raise; it returns the offender id list for logging.
    samples = _main_closed_non_traps()
    base = samples[0]
    diverged = base.model_copy(
        update={
            "world_config": base.world_config.model_copy(
                update={"issuer_behavior": {"settled_currency": "ZZZ"}}
            )
        }
    )
    offenders = safe_completion_spotcheck([diverged], enforce=False, split="main")
    assert diverged.sample_id in [o.split(":")[0] for o in offenders]


def test_gate_does_not_enforce_off_enforced_splits():
    # A divergence on a non-enforced split (lite) must not raise when enforce flows through
    # COVERAGE_ENFORCED_SPLITS membership (mirrors f1/coverage/solvability wiring).
    assert "lite" not in COVERAGE_ENFORCED_SPLITS
    samples = _main_closed_non_traps()
    base = samples[0]
    diverged = base.model_copy(
        update={
            "world_config": base.world_config.model_copy(
                update={"issuer_behavior": {"settled_currency": "ZZZ"}}
            )
        }
    )
    offenders = safe_completion_spotcheck(
        [diverged], enforce=("lite" in COVERAGE_ENFORCED_SPLITS), split="lite"
    )
    assert offenders  # computed + returned, never raised


def test_gate_ignores_traps_and_non_closed_gold():
    # A trap (is_trap=True) WITH a divergence is fine — traps are SUPPOSED to configure
    # divergence. Only CLOSED-gold non-traps are gated.
    samples = _main_closed_non_traps()
    base = samples[0]
    trap = base.model_copy(
        update={
            "is_trap": True,
            "world_config": base.world_config.model_copy(
                update={"issuer_behavior": {"settled_currency": "ZZZ"}}
            ),
        }
    )
    assert safe_completion_spotcheck([trap], enforce=True, split="main") == []
