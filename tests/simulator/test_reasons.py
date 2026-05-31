import pytest

from spar.simulator.reasons import (
    REASONS,
    DeclinePolicy,
    ReasonCode,
    classify_decline,
    correct_behaviors,
    is_decline,
    is_hard,
    retry_penalty_weight,
)


def test_all_ten_codes_present_with_full_metadata():
    expected = {"51", "65", "91", "05", "14", "54", "43", "46", "62", "1A"}
    assert set(REASONS) == expected
    for code, rc in REASONS.items():
        assert isinstance(rc, ReasonCode)
        assert rc.code == code
        assert rc.label
        assert rc.visa_category in {1, 2, 3, 4, None}
        assert isinstance(rc.correct_behaviors, tuple)
        assert rc.correct_behaviors  # non-empty


def test_one_a_is_an_auth_challenge_not_a_decline():
    # module 10 §6: 1A is a 3DS/SCA step-up, surfaced as requires_action.
    assert is_decline("1A") is False
    assert is_hard("1A") is False
    assert REASONS["1A"].cls == "auth_challenge"
    assert REASONS["1A"].visa_category is None
    assert REASONS["1A"].retryable is False


def test_real_declines_report_as_declines():
    for code in ("51", "65", "91", "05", "14", "54", "43", "46", "62"):
        assert is_decline(code) is True


def test_hard_codes_are_cat1_and_cat3_correctables_are_not_hard_blocks():
    # Cat-1 hard blocks: abort, never retry.
    for code in ("43", "46", "62"):
        assert is_hard(code) is True
        assert REASONS[code].visa_category == 1
        assert REASONS[code].retryable is False
    # Cat-3 correctables are "hard (correctable)" but recover by fixing data, not aborting.
    for code in ("14", "54"):
        assert REASONS[code].visa_category == 3
        assert REASONS[code].retryable is True
    # Soft Cat-2 retryable.
    for code in ("51", "65", "91"):
        assert REASONS[code].visa_category == 2
        assert is_hard(code) is False
        assert REASONS[code].retryable is True
    # Ambiguous Cat-4.
    assert REASONS["05"].visa_category == 4
    assert REASONS["05"].cls == "ambiguous"


def test_14_is_correct_card_data_not_account_updater():
    # module 10 §6 + axis §4: 14 fixes a typo'd PAN; 54 uses Account Updater.
    assert "correct_card_data" in correct_behaviors("14")
    assert "account_updater" not in correct_behaviors("14")
    assert "account_updater" in correct_behaviors("54")
    assert "correct_card_data" not in correct_behaviors("54")


def test_retry_penalty_weight_cat1_strictly_exceeds_cat4():
    # Retrying a Cat-1 hard decline costs more than a borderline Cat-4 reattempt.
    assert retry_penalty_weight("43") > retry_penalty_weight("05")
    assert retry_penalty_weight("46") > retry_penalty_weight("05")
    assert retry_penalty_weight("62") > retry_penalty_weight("05")
    # 1A is excluded from retry-penalty accounting entirely.
    assert retry_penalty_weight("1A") == 0.0
    # Soft Cat-2 retries are not free but cheaper than Cat-1.
    assert retry_penalty_weight("43") > retry_penalty_weight("51")


def test_retry_fee_risk_flags_cat1_codes():
    for code in ("43", "46", "62"):
        assert REASONS[code].retry_fee_risk is True
    assert REASONS["1A"].retry_fee_risk is False


def test_hard_decline_is_do_not_retry():
    assert classify_decline("43") is DeclinePolicy.HARD  # stolen_card (cls="hard")


def test_soft_decline_allows_bounded_retry():
    assert classify_decline("51") is DeclinePolicy.SOFT  # insufficient_funds (cls="soft")


def test_step_up_required_is_escalate():
    assert classify_decline("1A") is DeclinePolicy.STEP_UP  # auth challenge (cls="auth_challenge")


def test_hard_correctable_and_ambiguous_fold_into_soft():
    # C15: only cls=="hard" -> HARD; hard_correctable/ambiguous -> SOFT (bounded retry).
    assert classify_decline("14") is DeclinePolicy.SOFT   # hard_correctable
    assert classify_decline("05") is DeclinePolicy.SOFT   # ambiguous


@pytest.mark.parametrize("code", sorted(REASONS.keys()))
def test_every_reason_code_classifies_without_falling_through(code):
    # No emitted code may fall through unclassified (C15). Cross-check against the source of truth.
    result = classify_decline(code)
    assert isinstance(result, DeclinePolicy)
    cls = REASONS[code].cls
    expected = (
        DeclinePolicy.HARD if cls == "hard"
        else DeclinePolicy.STEP_UP if cls == "auth_challenge"
        else DeclinePolicy.SOFT
    )
    assert result is expected
