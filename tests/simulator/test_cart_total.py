"""Cart is priced from WorldConfig.cart_total (the honest in-scope cart), not amount_limit.

Bug (pre-fix): the world initialized the live cart price from `mandate.amount_limit` (the ceiling
= per_txn_max + $50), so the agent was SHOWN — and asked to authorize — a cart $50 over per_txn_max
on every in-scope sample, while gold said complete. Real LLMs then (correctly, per policy) escalated.
The generator now persists the intended cart total in `cart_total`, which the world prices from.
"""

from __future__ import annotations

from decimal import Decimal

from spar.dataset.generator import GenSpec, generate
from spar.simulator.enums import Axis, Difficulty, IntentSpec
from spar.simulator.world import World


def test_explicit_ok_cart_is_under_per_txn_max() -> None:
    """An in-scope (non-trap) explicit sample shows a cart at/under per_txn_max — so a policy-
    following agent can complete it instead of escalating an apparently-over-limit cart."""
    s = generate(GenSpec(axis=Axis.COMPLIANCE_TAX, seed=3, difficulty=Difficulty.EASY,
                         is_trap=False, intent_spec=IntentSpec.EXPLICIT))
    assert s.world_config.cart_total is not None
    w = World(s, trial_index=0)
    w.reset()
    obs = w.observe()
    assert obs.cart.subtotal == s.world_config.cart_total          # priced from cart_total
    assert obs.cart.subtotal <= s.mandate.per_txn_max              # genuinely in scope
    assert obs.cart.subtotal < s.mandate.amount_limit             # and under the ceiling


def test_world_falls_back_to_amount_limit_without_cart_total() -> None:
    """Legacy/hand-authored samples with no cart_total keep the old behavior (cart = amount_limit)
    so the fix is backward-compatible for the redline gold backbone."""
    s = generate(GenSpec(axis=Axis.COMPLIANCE_TAX, seed=3, difficulty=Difficulty.EASY,
                         is_trap=False, intent_spec=IntentSpec.EXPLICIT))
    legacy = s.model_copy(update={
        "world_config": s.world_config.model_copy(update={"cart_total": None})
    })
    w = World(legacy, trial_index=0)
    w.reset()
    assert w.observe().cart.subtotal == (s.mandate.amount_limit or Decimal("0"))
