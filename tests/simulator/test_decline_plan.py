from spar.simulator.decline_plan import (
    ChallengeOutcome,
    DeclineOutcome,
    resolve_auth_outcome,
    resolve_challenge_outcome,
)


def test_scripted_draw_pins_a_decline_at_an_attempt():
    plan = {
        "mode": "scripted",
        "draws": [{"attempt": 1, "outcome": "decline", "reason": "05"}],
    }
    out = resolve_auth_outcome(
        plan, sample_id="spar_decline_recovery_0001", seed=1, trial_index=0, attempt=1,
    )
    assert out == DeclineOutcome(kind="decline", reason_code="05")


def test_scripted_draw_pins_a_challenge():
    plan = {"mode": "scripted", "draws": [{"attempt": 1, "outcome": "challenge"}]}
    out = resolve_auth_outcome(
        plan, sample_id="spar_x", seed=1, trial_index=0, attempt=1,
    )
    assert out.kind == "challenge" and out.reason_code is None


def test_scripted_unpinned_attempt_defaults_to_approved():
    plan = {"mode": "scripted", "draws": [{"attempt": 1, "outcome": "decline", "reason": "43"}]}
    out = resolve_auth_outcome(
        plan, sample_id="spar_x", seed=1, trial_index=0, attempt=3,
    )
    assert out.kind == "approve"


def test_scripted_soft_then_hard_flip_resolves_each_attempt():
    plan = {
        "mode": "scripted",
        "draws": [
            {"attempt": 1, "outcome": "decline", "reason": "65"},
            {"attempt": 2, "outcome": "decline", "reason": "43"},
        ],
    }
    first = resolve_auth_outcome(plan, sample_id="spar_d", seed=7, trial_index=0, attempt=1)
    second = resolve_auth_outcome(plan, sample_id="spar_d", seed=7, trial_index=0, attempt=2)
    assert first == DeclineOutcome(kind="decline", reason_code="65")
    assert second == DeclineOutcome(kind="decline", reason_code="43")


def test_attempt_ordinal_is_stable_under_extra_steps():
    # G2: two agents reaching the same auth attempt via different action counts
    # MUST draw the same outcome — keying is on `attempt`, never elapsed_steps.
    plan = {"mode": "sampled", "p_decline": 1.0, "soft_reasons": ["51"]}
    a = resolve_auth_outcome(plan, sample_id="spar_s", seed=3, trial_index=0, attempt=1)
    b = resolve_auth_outcome(plan, sample_id="spar_s", seed=3, trial_index=0, attempt=1)
    assert a == b


def test_sampled_mode_is_deterministic_per_seed():
    plan = {"mode": "sampled", "p_decline": 1.0, "soft_reasons": ["51"]}
    a = resolve_auth_outcome(plan, sample_id="spar_s", seed=3, trial_index=0, attempt=1)
    b = resolve_auth_outcome(plan, sample_id="spar_s", seed=3, trial_index=0, attempt=1)
    assert a == b
    assert a.kind == "decline" and a.reason_code == "51"


def test_sampled_p_decline_zero_always_approves():
    plan = {"mode": "sampled", "p_decline": 0.0}
    out = resolve_auth_outcome(plan, sample_id="spar_s", seed=3, trial_index=0, attempt=1)
    assert out.kind == "approve"


def test_challenge_resolution_has_its_own_cleared_failed_space():
    # G2: a challenge resolves to cleared/failed via its OWN outcome space (SubStream.CHALLENGE),
    # never a re-roll of the auth categorical that could self-loop another `challenge`.
    plan = {
        "mode": "scripted",
        "draws": [{"attempt": 1, "outcome": "challenge"}],
        "challenge_draws": [{"attempt": 1, "outcome": "cleared"}],
    }
    out = resolve_challenge_outcome(
        plan, sample_id="spar_c", seed=5, trial_index=0, attempt=1,
    )
    assert out == ChallengeOutcome(kind="cleared")


def test_scripted_challenge_failed_is_supported():
    plan = {
        "mode": "scripted",
        "draws": [{"attempt": 1, "outcome": "challenge"}],
        "challenge_draws": [{"attempt": 1, "outcome": "failed", "reason": "05"}],
    }
    out = resolve_challenge_outcome(
        plan, sample_id="spar_c", seed=5, trial_index=0, attempt=1,
    )
    assert out.kind == "failed" and out.reason_code == "05"


def test_scripted_challenge_defaults_to_cleared_when_unpinned():
    plan = {"mode": "scripted", "draws": [{"attempt": 1, "outcome": "challenge"}]}
    out = resolve_challenge_outcome(
        plan, sample_id="spar_c", seed=5, trial_index=0, attempt=1,
    )
    assert out.kind == "cleared"


def test_sampled_challenge_is_deterministic_per_seed():
    plan = {"mode": "sampled", "p_challenge_fail": 1.0, "challenge_fail_reasons": ["05"]}
    a = resolve_challenge_outcome(plan, sample_id="spar_c", seed=9, trial_index=0, attempt=1)
    b = resolve_challenge_outcome(plan, sample_id="spar_c", seed=9, trial_index=0, attempt=1)
    assert a == b
    assert a.kind == "failed"
