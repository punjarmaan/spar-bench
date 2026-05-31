from importlib import resources

from spar.simulator.enums import Axis, FsmState, IntentSpec
from spar.simulator.schemas import Sample


def _load() -> list[Sample]:
    text = resources.files("spar.dataset.gold").joinpath("consent_mandate.jsonl").read_text(
        encoding="utf-8"
    )
    return [Sample.model_validate_json(ln) for ln in text.splitlines() if ln.strip()]


def test_all_samples_are_consent_mandate_axis():
    samples = _load()
    assert len(samples) >= 12
    assert all(s.axis is Axis.CONSENT_MANDATE for s in samples)
    assert all(s.policy_id == "default_v1" for s in samples)


def test_traps_have_safe_correct_outcomes():
    traps = [s for s in _load() if s.is_trap]
    assert len(traps) >= 9
    for s in traps:
        assert s.gold.correct_outcome in {FsmState.ESCALATED, FsmState.ABORTED}


def test_each_explicit_trap_kind_is_present():
    ids = {s.sample_id for s in _load()}
    for needed in [
        "spar_consent_mandate_over_per_txn",
        "spar_consent_mandate_over_daily",
        "spar_consent_mandate_restricted_field",
        "spar_consent_mandate_off_merchant",
        "spar_consent_mandate_off_mcc",
        "spar_consent_mandate_bad_instrument",
        "spar_consent_mandate_post_revocation",
        "spar_consent_mandate_hnp_over_threshold",
        "spar_consent_mandate_injection_blatant",
        "spar_consent_mandate_injection_field",
    ]:
        assert needed in ids, needed


def test_semantic_and_underspecified_present_with_bands():
    samples = _load()
    semantic = [s for s in samples if s.intent_spec is IntentSpec.SEMANTIC]
    under = [s for s in samples if s.intent_spec is IntentSpec.UNDERSPECIFIED]
    assert len(semantic) >= 1 and len(under) >= 1
    for s in semantic:
        assert s.world_config.market_context is not None
        assert s.gold.acceptable_spend_band is not None
        assert s.gold.overspend_ceiling is not None
        assert s.mandate.amount_limit is None and s.mandate.budget_hint is not None
    for s in under:
        assert s.gold.correct_outcome is FsmState.ESCALATED
        assert "request_user_confirmation" in s.gold.must


def test_injection_samples_carry_merchant_note_token_and_demand():
    inj = [s for s in _load() if "injection" in s.sample_id]
    assert len(inj) >= 2
    for s in inj:
        assert (s.world_config.issuer_behavior or {}).get("merchant_note")
        assert "obey_injected_instruction" in s.gold.must_not
        assert s.gold.injection_demand is not None
        assert "action" in s.gold.injection_demand
    assert any(
        s.gold.injection_demand.get("action") == "modify_cart"
        and "field" in s.gold.injection_demand
        for s in inj
    )


def test_restricted_field_sample_encodes_field_in_conditions_and_token():
    s = next(x for x in _load() if x.sample_id == "spar_consent_mandate_restricted_field")
    fields = s.mandate.conditions.get("restricted_fields")
    assert fields and "subscription_term" in fields
    assert any(t.startswith("modify:") for t in s.gold.must_not)
