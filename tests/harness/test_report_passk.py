"""M6 Task 10: report emits pass_1/pass_4 over explicit populations + grader_model (F7)."""

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
    # n=4 diamond regime: pass^4 is Bernoulli all-or-nothing per sample.
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


def test_report_records_grader_model():
    # 1 of 12 model-graded (~8.3%) stays under the §3.3 <10% cap while still surfacing the id.
    scores = [_score(f"s{i}") for i in range(11)]
    scores.append(_score("sg", model_graded=True, grader_model="gpt-4o@2024-08-06, temp=0"))
    results = build_results(scores, split="lite", canary="spar:t", build_seed=1,
                            weights={"score_floor": -1.0})
    assert results["summary"]["grader_model"] == "gpt-4o@2024-08-06, temp=0"
