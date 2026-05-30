"""EM3 consolidation behavior tests (offline, golden-fixture driven)."""

from __future__ import annotations

import json
from pathlib import Path

from spar.eval.consolidate import (
    AXES_ORDER,
    INTENT_SPECS_ORDER,
    LeaderboardEntry,
    consolidate,
    trust_ci95,
    write_leaderboard,
)


def _make_entry(**overrides: object) -> LeaderboardEntry:
    base: dict[str, object] = {
        "model": "opus-frontier", "cls": "frontier",
        "trust_score": 0.71, "trust_score_ci95": (0.67, 0.75),
        "trust_score_objective": 0.70, "overspend_rate": 0.03,
        "false_refusal_rate": 0.09, "pass_1": 0.74, "pass_4": 0.62,
        "axes": {a: 0.7 for a in AXES_ORDER},
        "by_intent_spec": {i: 0.6 for i in INTENT_SPECS_ORDER},
        "n_main": 8, "n_diamond": 6, "scored_fraction": 1.0, "status": "verified",
        "cost_usd": 4.10, "provenance": "private_verified",
        "model_version_pin": "anthropic/claude-opus-4@2026-xx",
        "scaffold_version": "1.0.0", "spar_version": "0.1.0",
        "dataset_canary": "spar:abc", "run_date": "2026-05-29"}
    base.update(overrides)
    return LeaderboardEntry(**base)  # type: ignore[arg-type]


def test_entry_serializes_cls_as_class_alias() -> None:
    entry = _make_entry()
    dumped = entry.model_dump(by_alias=True)
    assert dumped["class"] == "frontier"     # serialized key is "class" (Python attr is `cls`)
    assert "cls" not in dumped
    assert dumped["trust_score_ci95"] == [0.67, 0.75]   # tuple serializes to a 2-element list
    assert dumped["cost_usd"] == 4.10


def test_entry_round_trips_from_class_alias() -> None:
    payload = {"class": "open", "model": "llama-open", "trust_score": 0.21,
               "trust_score_ci95": (0.18, 0.24), "trust_score_objective": 0.20,
               "overspend_rate": 0.40, "false_refusal_rate": 0.05, "pass_1": 0.30,
               "pass_4": 0.10, "axes": {a: 0.2 for a in AXES_ORDER},
               "by_intent_spec": {i: 0.15 for i in INTENT_SPECS_ORDER},
               "n_main": 5, "n_diamond": 6, "scored_fraction": 0.90, "status": "partial",
               "cost_usd": 0.15, "provenance": "public_self_run",
               "model_version_pin": "meta-llama/llama-3-70b@2026-xx",
               "scaffold_version": "1.0.0", "spar_version": "0.1.0",
               "dataset_canary": "spar:abc", "run_date": "2026-05-29"}
    entry = LeaderboardEntry.model_validate(payload)
    assert entry.cls == "open"
    assert entry.status == "partial"


def test_trust_ci95_is_ordered_and_within_bounds() -> None:
    scores = [0.7, 0.8, 0.6, 0.9, 0.7, 0.65, 0.75, 0.72]
    lo, hi = trust_ci95(scores)
    assert 0.0 <= lo <= hi <= 1.0
    mean = sum(scores) / len(scores)
    assert lo <= mean <= hi           # the sample mean lies inside its own 95% CI


def test_trust_ci95_is_deterministic_across_two_calls() -> None:
    scores = [0.7, 0.8, 0.6, 0.9, 0.7, 0.65, 0.75, 0.72]
    assert trust_ci95(scores) == trust_ci95(scores)     # byte-identical (seeded RNG)


def test_trust_ci95_clamps_out_of_range_scores() -> None:
    # Scores below 0 / above 1 are clamped before resampling (spec §5.8 "clamped scores").
    lo, hi = trust_ci95([-0.5, 1.5, 0.5])
    assert 0.0 <= lo <= hi <= 1.0


def test_trust_ci95_degenerate_single_score() -> None:
    lo, hi = trust_ci95([0.5])
    assert lo == hi == 0.5            # every resample is the same single value


def test_trust_ci95_empty_is_zero_zero() -> None:
    assert trust_ci95([]) == (0.0, 0.0)


def test_consolidate_single_model_pulls_competence_from_main(single_runs_dir: Path) -> None:
    entries = consolidate(single_runs_dir)
    assert len(entries) == 1
    e = entries[0]
    assert e.model == "opus-frontier"
    assert e.cls == "frontier"
    assert e.trust_score == 0.71                  # summary.trust_score from main
    assert e.trust_score_objective == 0.70        # summary.trust_score_objective from main
    assert e.overspend_rate == 0.03
    assert e.false_refusal_rate == 0.09
    assert e.pass_1 == 0.74                        # main pass^1
    assert e.pass_4 == 0.62                        # diamond pass^4
    assert e.n_main == 8                           # main summary.n_samples
    assert e.n_diamond == 6                        # diamond summary.n_samples


def test_consolidate_pulls_axes_and_intents_from_main(single_runs_dir: Path) -> None:
    e = consolidate(single_runs_dir)[0]
    assert e.axes == {"routing": 0.81, "decline_recovery": 0.74, "consent_mandate": 0.69,
                      "stale_state": 0.70, "compliance_tax": 0.78, "fraud_reactivity": 0.66,
                      "post_purchase": 0.71}
    assert list(e.axes.keys()) == AXES_ORDER       # canonical ordering preserved
    assert e.by_intent_spec == {"explicit": 0.75, "semantic": 0.61, "underspecified": 0.55}
    assert list(e.by_intent_spec.keys()) == INTENT_SPECS_ORDER


def test_consolidate_computes_ci_from_main_per_sample(single_runs_dir: Path) -> None:
    e = consolidate(single_runs_dir)[0]
    expected = trust_ci95([0.7, 0.8, 0.6, 0.9, 0.7, 0.65, 0.75, 0.72])  # opus main per_sample
    assert e.trust_score_ci95 == expected


def test_consolidate_pulls_version_pins_and_canary(single_runs_dir: Path) -> None:
    e = consolidate(single_runs_dir)[0]
    assert e.model_version_pin == "anthropic/claude-opus-4@2026-xx"
    assert e.scaffold_version == "1.0.0"
    assert e.spar_version == "0.1.0"
    assert e.dataset_canary == "spar:00000000-0000-0000-0000-000000000000"
    assert e.run_date == "2026-05-29"


def _by_model(runs: Path) -> dict[str, LeaderboardEntry]:
    return {e.model: e for e in consolidate(runs)}


def test_full_run_is_verified(runs_dir: Path) -> None:
    opus = _by_model(runs_dir)["opus-frontier"]
    assert opus.scored_fraction == 1.0
    assert opus.status == "verified"


def test_partial_run_below_floor_is_partial(runs_dir: Path) -> None:
    llama = _by_model(runs_dir)["llama-open"]
    assert llama.scored_fraction == 0.90        # scored_main=9 / total_main=10
    assert llama.status == "partial"            # 0.90 < PUBLISHABILITY_FLOOR (0.98)


def test_provenance_private_verified_from_manifest(runs_dir: Path) -> None:
    opus = _by_model(runs_dir)["opus-frontier"]
    assert opus.provenance == "private_verified"
    assert opus.cost_usd == 4.10               # agent inference only (from manifest)


def test_provenance_public_self_run_from_manifest(runs_dir: Path) -> None:
    llama = _by_model(runs_dir)["llama-open"]
    assert llama.provenance == "public_self_run"
    assert llama.cost_usd == 0.15


def test_provenance_defaults_public_when_absent(single_runs_dir: Path) -> None:
    # Strip provenance from the manifest -> default to public_self_run (spec §5.9).
    manifest_path = single_runs_dir / "opus-frontier" / "run_manifest.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    del data["provenance"]
    manifest_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    e = consolidate(single_runs_dir)[0]
    assert e.provenance == "public_self_run"
