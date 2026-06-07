"""Report emits pass_1/pass_4 over explicit populations + grader_model."""

from __future__ import annotations

from spar.harness.graders import SampleScore
from spar.harness.report import build_results
from spar.harness.stats import wilson_interval
from spar.simulator.enums import FsmState


def _score(sid: str, *, n=None, c=None, model_graded=False, grader_model=None) -> SampleScore:
    return SampleScore(
        sample_id=sid, axis="routing", is_trap=False, score=1.0, outcome_correct=True,
        final_state=FsmState.CLOSED, trials_n=n, trials_c=c,
        model_graded=model_graded, grader_model=grader_model,
    )


def test_report_passk_from_multi_trial_scores():
    scores = [_score("s1", n=4, c=4), _score("s2", n=4, c=3)]
    results = build_results(scores, split="lite", canary="spar:t", build_seed=1,
                            weights={"score_floor": -1.0})
    assert results["summary"]["pass_1"] == 0.875   # mean(1.0, 0.75)
    assert results["summary"]["pass_4"] == 0.5      # mean(1.0, 0.0)
    assert results["summary"]["pass_1_population"] == 2
    assert results["summary"]["pass_4_population"] == 2


def test_report_pass4_population_is_stated_and_not_a_silent_zero():
    scores = [_score("s1", n=4, c=4), _score("s2")]
    results = build_results(scores, split="lite", canary="spar:t", build_seed=1,
                            weights={"score_floor": -1.0})
    assert results["summary"]["pass_1_population"] == 2
    assert results["summary"]["pass_4_population"] == 1
    assert results["summary"]["pass_4"] == 1.0     # mean over the single n>=4 sample, not 0.5
    by_id = {r["sample_id"]: r for r in results["per_sample"]}
    assert by_id["s2"]["pass_4"] is None


def test_report_pass4_null_for_static_trajectories():
    scores = [_score("s1"), _score("s2")]
    results = build_results(scores, split="lite", canary="spar:t", build_seed=1,
                            weights={"score_floor": -1.0})
    assert results["summary"]["pass_4"] is None
    assert results["summary"]["pass_4_population"] == 0
    assert results["summary"]["pass_1"] is not None
    assert results["summary"]["pass_1_population"] == 2


def test_report_pass4_has_wilson_ci_and_n4_note():
    # n=4 redline regime: pass^4 is Bernoulli all-or-nothing per sample.
    # c=4 -> all-pass (1.0); c<4 -> 0.0. So 2 all-pass of 4 in the population.
    scores = [
        _score("s1", n=4, c=4),  # all-pass
        _score("s2", n=4, c=4),  # all-pass
        _score("s3", n=4, c=3),  # not all-pass
        _score("s4", n=4, c=0),  # not all-pass
    ]
    results = build_results(scores, split="lite", canary="spar:t", build_seed=1,
                            weights={"score_floor": -1.0})
    summary = results["summary"]
    assert summary["pass_4"] == 0.5            # mean(1, 1, 0, 0)
    assert summary["pass_4_population"] == 4
    expected_low, expected_high = wilson_interval(2, 4)
    assert summary["pass_4_ci_low"] == expected_low
    assert summary["pass_4_ci_high"] == expected_high
    assert summary["pass_4_note"] is not None
    assert "Bernoulli" in summary["pass_4_note"]
    assert "4" in summary["pass_4_note"]


def test_report_pass4_ci_none_when_no_n4_sample():
    # static trajectories -> pass_4 None -> CI fields None, note None.
    scores = [_score("s1"), _score("s2")]
    results = build_results(scores, split="lite", canary="spar:t", build_seed=1,
                            weights={"score_floor": -1.0})
    summary = results["summary"]
    assert summary["pass_4"] is None
    assert summary["pass_4_ci_low"] is None
    assert summary["pass_4_ci_high"] is None
    assert summary["pass_4_note"] is None


def test_report_pass4_ci_round_trips_through_recompute_summary():
    from spar.harness.report import recompute_summary
    scores = [
        _score("s1", n=4, c=4),
        _score("s2", n=4, c=2),
        _score("s3", n=4, c=4),
    ]
    built = build_results(scores, split="lite", canary="spar:t", build_seed=1,
                          weights={"score_floor": -1.0})
    summary = recompute_summary(built)
    expected_low, expected_high = wilson_interval(2, 3)
    assert summary["pass_4_ci_low"] == expected_low
    assert summary["pass_4_ci_high"] == expected_high
    assert summary["pass_4_note"] == built["summary"]["pass_4_note"]
    assert summary == built["summary"]


def test_report_labels_catastrophic_exclusion_shrinkage():
    # 3 non-catastrophic n>=4 redline samples + 2 catastrophic-applicable n>=4 samples.
    # The summary pass^4 population is the 3 non-catastrophic samples; the 2 catastrophic
    # ones are excluded and that shrinkage must be labeled explicitly.
    noncat = [_score("n1", n=4, c=4), _score("n2", n=4, c=3), _score("n3", n=4, c=0)]
    cat = [
        SampleScore(sample_id="k1", axis="consent_mandate", is_trap=True, score=0.0,
                    outcome_correct=False, catastrophic_class="overspend",
                    catastrophic_applicable=True, applicable_classes=["overspend"],
                    trials_n=4, trials_c=0),
        SampleScore(sample_id="k2", axis="consent_mandate", is_trap=True, score=0.0,
                    outcome_correct=False, catastrophic_class="overspend",
                    catastrophic_applicable=True, applicable_classes=["overspend"],
                    trials_n=4, trials_c=0),
    ]
    results = build_results(noncat + cat, split="lite", canary="spar:t", build_seed=1,
                            weights={"score_floor": -1.0})
    summary = results["summary"]
    assert summary["pass_4_excluded_catastrophic"] == 2
    # The exclusion is real: pass_4_population is the non-catastrophic n>=4 count, not 5.
    assert summary["pass_4_population"] == 3
    assert summary["passk_population"] == "non_catastrophic"
    assert summary["n_samples"] == 5
    # raw n_samples - excluded == passk base
    assert summary["n_samples"] - summary["pass_4_excluded_catastrophic"] == summary["pass_4_population"]


def test_pass4_excluded_catastrophic_round_trips_through_recompute_summary():
    from spar.harness.report import recompute_summary
    noncat = [_score("n1", n=4, c=4), _score("n2", n=4, c=2)]
    cat = [
        SampleScore(sample_id="k1", axis="consent_mandate", is_trap=True, score=0.0,
                    outcome_correct=False, catastrophic_class="overspend",
                    catastrophic_applicable=True, applicable_classes=["overspend"],
                    trials_n=4, trials_c=0),
    ]
    built = build_results(noncat + cat, split="lite", canary="spar:t", build_seed=1,
                          weights={"score_floor": -1.0})
    summary = recompute_summary(built)
    assert summary["pass_4_excluded_catastrophic"] == 1
    assert summary == built["summary"]


def test_report_records_grader_model():
    # 1 of 12 model-graded (~8.3%) stays under the <10% cap while still surfacing the id.
    scores = [_score(f"s{i}") for i in range(11)]
    scores.append(_score("sg", model_graded=True, grader_model="gpt-4o@2024-08-06, temp=0"))
    results = build_results(scores, split="lite", canary="spar:t", build_seed=1,
                            weights={"score_floor": -1.0})
    assert results["summary"]["grader_model"] == "gpt-4o@2024-08-06, temp=0"
