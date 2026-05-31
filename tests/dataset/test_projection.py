from spar.simulator.enums import Axis, Difficulty, IntentSpec
from spar.dataset.generator import GenSpec, generate
from spar.dataset.projection import public_view, HIDDEN_WORLD_CONFIG_KEYS


def _semantic_sample():
    # semantic => world_config carries the HIDDEN market_context percentiles.
    return generate(GenSpec(axis=Axis.CONSENT_MANDATE, seed=5, difficulty=Difficulty.MEDIUM,
                            is_trap=False, intent_spec=IntentSpec.SEMANTIC))


def _catastrophic_sample():
    # a consent_mandate trap populates the hidden issuer_behavior (the catastrophic answer).
    return generate(GenSpec(axis=Axis.CONSENT_MANDATE, seed=1, difficulty=Difficulty.EASY,
                            is_trap=True, intent_spec=IntentSpec.EXPLICIT))


def test_public_view_of_catastrophic_sample_strips_gold_and_issuer_behavior():
    s = _catastrophic_sample()
    assert s.world_config.issuer_behavior  # precondition: it IS a catastrophic-applicable sample
    pv = public_view(s)
    assert "gold" not in pv
    assert "world_config" not in pv
    assert "issuer_behavior" not in repr(pv)
    assert pv["mandate"]["amount_limit"] is not None


def test_public_view_strips_the_entire_gold_block():
    pv = public_view(_semantic_sample())
    assert "gold" not in pv


def test_public_view_strips_all_hidden_world_config_fields():
    pv = public_view(_semantic_sample())
    methods = pv["methods"]
    for m in methods:
        # only advertised fee + the coarse band are visible; never the hidden truth.
        assert set(m) <= {"acquirer_id", "methods", "geos",
                          "advertised_fee_bps", "observed_approval_band"}
        assert "true_fee_bps" not in m
        assert "approval_prob" not in m
        assert "reliability" not in m
    # market_context (the grader's ground truth) must not leak anywhere.
    assert "market_context" not in pv
    flat = repr(pv)
    for key in HIDDEN_WORLD_CONFIG_KEYS:
        assert key not in flat, f"hidden world_config key {key!r} leaked into public_view"


def test_public_view_keeps_only_agent_visible_top_level_fields():
    pv = public_view(_semantic_sample())
    assert set(pv) == {"sample_id", "axis", "difficulty", "policy_id", "canary",
                       "seed", "mandate", "cart", "methods"}


def test_public_view_contains_no_hidden_or_gold_key_property():
    # Property: across many generated samples + every axis/intent, no hidden/gold
    # key ever appears in the projection (shared with the module-10 §3.1 leak test).
    banned = set(HIDDEN_WORLD_CONFIG_KEYS) | {
        "gold", "correct_outcome", "oracle_route", "is_trap",
        "acceptable_spend_band", "overspend_ceiling", "must", "must_not",
        "injection_demand", "world_config", "true_fee_bps", "approval_prob",
        "reliability", "decline_plan",
    }
    for axis in Axis:
        for intent in IntentSpec:
            for trap in (True, False):
                if intent is not IntentSpec.EXPLICIT and trap:
                    continue
                s = generate(GenSpec(axis=axis, seed=7, difficulty=Difficulty.HARD,
                                     is_trap=trap, intent_spec=intent))
                flat = repr(public_view(s))
                for key in banned:
                    assert key not in flat, f"{axis}/{intent}: {key!r} leaked"
