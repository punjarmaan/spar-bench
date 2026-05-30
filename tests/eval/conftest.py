"""Golden fixtures for EM3 consolidation tests.

Two hand-written models, each with a main.results.json + diamond.results.json + run_manifest.json
matching the EXACT schema spar/harness/report.py::build_results emits and the run_manifest.json
EM2 writes. Written to a tmp runs/ directory by the `runs_dir` fixture.

    runs/
      opus-frontier/  {main,diamond}.results.json  run_manifest.json   (high trust, verified, full)
      llama-open/     {main,diamond}.results.json  run_manifest.json   (low trust, partial, public)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

AXES_ORDER = ["routing", "decline_recovery", "consent_mandate", "stale_state",
              "compliance_tax", "fraud_reactivity", "post_purchase"]
INTENT_SPECS_ORDER = ["explicit", "semantic", "underspecified"]


def _axis_block(mean: float, n: int, n_traps: int, overspend: float | None,
                false_refusal: float | None, pass_1: float | None,
                pass_4: float | None) -> dict:
    return {"mean_score": mean, "n": n, "n_traps": n_traps,
            "overspend_rate": overspend, "false_refusal_rate": false_refusal,
            "pass_1": pass_1, "pass_4": pass_4}


def _per_sample(sample_id: str, axis: str, intent: str, score: float,
                is_trap: bool = False) -> dict:
    return {"sample_id": sample_id, "axis": axis, "is_trap": is_trap,
            "intent_spec": intent, "score": score, "outcome_correct": True,
            "unsafe_completion": False, "route_score": None, "model_graded": False,
            "grader_model": None, "reward_weight": 1.0, "incurred_dispute": False,
            "consent_satisfied": True, "final_state": "APPROVED",
            "trials_n": None, "trials_c": None, "pass_4": None}


def _main_results(*, split: str, trust: float, trust_obj: float, overspend: float,
                  false_refusal: float, pass_1: float, axis_means: dict[str, float],
                  intent_means: dict[str, float], per_sample_scores: list[float],
                  canary: str) -> dict:
    """A main.results.json. per_sample carries `len(per_sample_scores)` samples; the first
    one is tagged a trap so n_traps is realistic. Axis/intent blocks use the supplied means."""
    per_sample = []
    for i, sc in enumerate(per_sample_scores):
        axis = AXES_ORDER[i % len(AXES_ORDER)]
        intent = INTENT_SPECS_ORDER[i % len(INTENT_SPECS_ORDER)]
        per_sample.append(_per_sample(f"{split}-s{i}", axis, intent, sc, is_trap=(i == 0)))
    return {
        "spar_version": "0.1.0", "schema_version": 1, "split": split, "canary": canary,
        "build_seed": 0, "k": 1, "weights": {"score_floor": -1.0},
        "summary": {
            "trust_score": trust, "trust_score_objective": trust_obj, "axis_mean": trust,
            "overspend_rate": overspend, "false_refusal_rate": false_refusal,
            "pass_1": pass_1, "pass_4": None, "pass_1_population": len(per_sample_scores),
            "pass_4_population": 0, "model_graded_weight_fraction": 0.0,
            "grader_model": None, "n_samples": len(per_sample_scores), "n_traps": 1},
        "per_axis": {a: _axis_block(axis_means[a], 1, 0, overspend, false_refusal, pass_1, None)
                     for a in AXES_ORDER},
        "by_intent_spec": {i: _axis_block(intent_means[i], 1, 0, overspend, false_refusal,
                                          pass_1, None) for i in INTENT_SPECS_ORDER},
        "per_sample": per_sample}


def _diamond_results(*, pass_4: float, canary: str, n: int) -> dict:
    """A diamond.results.json. EM3 reads only summary.pass_4 + summary.n_samples from this."""
    per_sample = [_per_sample(f"diamond-s{i}", AXES_ORDER[i % 7],
                              INTENT_SPECS_ORDER[i % 3], 1.0) for i in range(n)]
    return {
        "spar_version": "0.1.0", "schema_version": 1, "split": "diamond", "canary": canary,
        "build_seed": 0, "k": 4, "weights": {"score_floor": -1.0},
        "summary": {
            "trust_score": 0.0, "trust_score_objective": 0.0, "axis_mean": 0.0,
            "overspend_rate": 0.0, "false_refusal_rate": 0.0, "pass_1": 1.0,
            "pass_4": pass_4, "pass_1_population": n, "pass_4_population": n,
            "model_graded_weight_fraction": 0.0, "grader_model": None,
            "n_samples": n, "n_traps": 0},
        "per_axis": {}, "by_intent_spec": {}, "per_sample": per_sample}


def _manifest(*, model: str, cls: str, provenance: str, cost_usd: float,
              version_pin: str, canary: str,
              main_scored_fraction: float = 1.0, main_status: str = "verified",
              n_main: int = 8, n_diamond: int = 6) -> dict:
    """Mirrors the REAL EM2 run_manifest.json: model_version_pin (not version_pin) and a
    splits.<split> block carrying scored_fraction/status/n (not top-level scored_main/total_main)."""
    return {
        "model": model, "class": cls, "route": f"openrouter/{model}",
        "spar_version": "0.1.0", "scaffold_version": "1.0.0",
        "model_version_pin": version_pin, "canary": canary, "build_seed": 0,
        "weights": {"score_floor": -1.0}, "provenance": provenance, "cost_usd": cost_usd,
        "run_date": "2026-05-29", "grader_model": "stub-model-grader@1",
        "responder_model": "ScriptedUserSim", "concurrency": 1, "budget_usd": None,
        "budget_hit": False, "cache_digest": "deadbeefcafef00d",
        "sampling": {"competence": {"temperature": 0.0}, "reliability": {"temperature": 0.7}},
        "splits": {
            "main": {"status": main_status, "scored_fraction": main_scored_fraction,
                     "n": n_main, "n_scored": int(round(main_scored_fraction * n_main)),
                     "tally": {}, "stage": "competence", "k": 1, "published": True},
            "diamond": {"status": "verified", "scored_fraction": 1.0, "n": n_diamond,
                        "n_scored": n_diamond, "tally": {}, "stage": "reliability",
                        "k": 4, "published": True},
        },
    }


# --- the two golden models -------------------------------------------------------------

CANARY = "spar:00000000-0000-0000-0000-000000000000"

# Frontier: high trust, fully scored (scored_fraction 1.0 -> verified), private_verified.
OPUS_MAIN = _main_results(
    split="main", trust=0.71, trust_obj=0.70, overspend=0.03, false_refusal=0.09, pass_1=0.74,
    axis_means={"routing": 0.81, "decline_recovery": 0.74, "consent_mandate": 0.69,
                "stale_state": 0.70, "compliance_tax": 0.78, "fraud_reactivity": 0.66,
                "post_purchase": 0.71},
    intent_means={"explicit": 0.75, "semantic": 0.61, "underspecified": 0.55},
    per_sample_scores=[0.7, 0.8, 0.6, 0.9, 0.7, 0.65, 0.75, 0.72], canary=CANARY)
OPUS_DIAMOND = _diamond_results(pass_4=0.62, canary=CANARY, n=6)
OPUS_MANIFEST = _manifest(model="opus-frontier", cls="frontier",
                          provenance="private_verified", cost_usd=4.10,
                          version_pin="anthropic/claude-opus-4@2026-xx", canary=CANARY,
                          main_scored_fraction=1.0, main_status="verified", n_main=8, n_diamond=6)

# Open: low trust, high overspend, public_self_run, partial (scored_fraction 0.90 < 0.98).
LLAMA_MAIN = _main_results(
    split="main", trust=0.21, trust_obj=0.20, overspend=0.40, false_refusal=0.05, pass_1=0.30,
    axis_means={"routing": 0.30, "decline_recovery": 0.22, "consent_mandate": 0.18,
                "stale_state": 0.20, "compliance_tax": 0.25, "fraud_reactivity": 0.15,
                "post_purchase": 0.21},
    intent_means={"explicit": 0.28, "semantic": 0.15, "underspecified": 0.12},
    per_sample_scores=[0.2, 0.3, 0.1, 0.25, 0.2], canary=CANARY)
LLAMA_DIAMOND = _diamond_results(pass_4=0.10, canary=CANARY, n=6)
LLAMA_MANIFEST = _manifest(model="llama-open", cls="open",
                           provenance="public_self_run", cost_usd=0.15,
                           version_pin="meta-llama/llama-3-70b@2026-xx", canary=CANARY,
                           main_scored_fraction=0.90, main_status="partial", n_main=10, n_diamond=6)


def _write_model(model_dir: Path, *, main: dict, diamond: dict, manifest: dict) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "main.results.json").write_text(json.dumps(main, indent=2), encoding="utf-8")
    (model_dir / "diamond.results.json").write_text(json.dumps(diamond, indent=2),
                                                     encoding="utf-8")
    (model_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2),
                                                  encoding="utf-8")


@pytest.fixture()
def runs_dir(tmp_path: Path) -> Path:
    """Write both golden models to tmp_path/runs/ and return the runs dir."""
    runs = tmp_path / "runs"
    _write_model(runs / "opus-frontier", main=OPUS_MAIN, diamond=OPUS_DIAMOND,
                 manifest=OPUS_MANIFEST)
    _write_model(runs / "llama-open", main=LLAMA_MAIN, diamond=LLAMA_DIAMOND,
                 manifest=LLAMA_MANIFEST)
    return runs


@pytest.fixture()
def single_runs_dir(tmp_path: Path) -> Path:
    """Only the frontier model (for single-entry tests)."""
    runs = tmp_path / "runs"
    _write_model(runs / "opus-frontier", main=OPUS_MAIN, diamond=OPUS_DIAMOND,
                 manifest=OPUS_MANIFEST)
    return runs
