"""The published leaderboard headline is the refusal-aware trust_score_useful,
NOT the refusal-blind trust_score. Raw + objective are retained as secondary columns.

Mirrors tests/eval/conftest.py's golden-fixture layout (runs/<model>/{main,redline}.results.json
+ run_manifest.json), but supplies a summary where trust_score (0.80) and trust_score_useful (0.30)
DIFFER so we can prove which one is published.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from tests.eval.conftest import (
    AXES_ORDER,
    CANARY,
    INTENT_SPECS_ORDER,
    OPUS_REDLINE,
    _main_results,
    _manifest,
    _write_model,
)

from spar.eval.consolidate import consolidate, write_leaderboard

# A summary where the refusal-blind trust_score and the refusal-aware useful score diverge:
# trust_score=0.80, false_refusal_rate=0.625 -> trust_score_useful = 0.80 * (1 - 0.625) = 0.30.
RAW_TRUST = 0.80
USEFUL_TRUST = 0.30
FALSE_REFUSAL = 0.625


def _useful_main() -> dict:
    """A main.results.json whose summary carries BOTH trust_score and trust_score_useful,
    matching the real report.py summary keys (which the conftest helper omits)."""
    main = _main_results(
        split="main", trust=RAW_TRUST, trust_obj=0.70,
        unsafe_completion=0.03, false_refusal=FALSE_REFUSAL, pass_1=0.74,
        axis_means={a: 0.7 for a in AXES_ORDER},
        intent_means={i: 0.6 for i in INTENT_SPECS_ORDER},
        per_sample_scores=[0.7, 0.8, 0.6, 0.9, 0.7, 0.65, 0.75, 0.72], canary=CANARY)
    # report.py emits trust_score_useful in the summary; the conftest helper predates it.
    main["summary"]["trust_score_useful"] = USEFUL_TRUST
    return main


def _useful_runs(tmp_path: Path) -> Path:
    runs = tmp_path / "runs"
    manifest = _manifest(model="diverge-model", cls="frontier",
                         provenance="private_verified", cost_usd=2.0,
                         version_pin="x/diverge@2026", canary=CANARY,
                         main_scored_fraction=1.0, main_status="verified",
                         n_main=8, n_redline=6)
    _write_model(runs / "diverge-model", main=_useful_main(),
                 redline=OPUS_REDLINE, manifest=manifest)
    return runs


def test_headline_entry_field_is_useful_not_raw(tmp_path: Path) -> None:
    e = consolidate(_useful_runs(tmp_path))[0]
    # The headline/primary score is the refusal-aware useful score, NOT the raw refusal-blind one.
    assert e.trust_score == USEFUL_TRUST
    assert e.trust_score != RAW_TRUST
    # Raw refusal-blind score + objective are still present (nothing dropped).
    assert e.trust_score_raw == RAW_TRUST
    assert e.trust_score_objective == 0.70


def test_headline_csv_publishes_useful_and_retains_raw(tmp_path: Path) -> None:
    out = tmp_path / "out"
    write_leaderboard(consolidate(_useful_runs(tmp_path)), out)
    with (out / "leaderboard.csv").open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        header = reader.fieldnames or []
        rows = list(reader)
    assert "trust_score" in header          # headline column
    assert "trust_score_raw" in header      # retained refusal-blind diagnostic
    assert "trust_score_objective" in header
    row = rows[0]
    assert float(row["trust_score"]) == USEFUL_TRUST       # headline = useful
    assert float(row["trust_score_raw"]) == RAW_TRUST      # raw retained, distinguishable
    assert float(row["trust_score_objective"]) == 0.70


def test_headline_sort_is_by_useful_score(tmp_path: Path) -> None:
    # Two private_verified models: A has higher RAW but lower USEFUL than B.
    # Sorting by the useful headline must rank B above A.
    runs = tmp_path / "runs"
    a_main = _main_results(
        split="main", trust=0.90, trust_obj=0.70, unsafe_completion=0.03,
        false_refusal=0.7, pass_1=0.74, axis_means={x: 0.7 for x in AXES_ORDER},
        intent_means={i: 0.6 for i in INTENT_SPECS_ORDER},
        per_sample_scores=[0.7, 0.8], canary=CANARY)
    a_main["summary"]["trust_score_useful"] = 0.27        # 0.90 * (1 - 0.7)
    b_main = _main_results(
        split="main", trust=0.50, trust_obj=0.70, unsafe_completion=0.03,
        false_refusal=0.1, pass_1=0.74, axis_means={x: 0.7 for x in AXES_ORDER},
        intent_means={i: 0.6 for i in INTENT_SPECS_ORDER},
        per_sample_scores=[0.7, 0.8], canary=CANARY)
    b_main["summary"]["trust_score_useful"] = 0.45        # 0.50 * (1 - 0.1)
    for name, m in (("a-high-raw", a_main), ("b-high-useful", b_main)):
        manifest = _manifest(model=name, cls="frontier", provenance="private_verified",
                             cost_usd=1.0, version_pin=f"x/{name}@1", canary=CANARY)
        _write_model(runs / name, main=m, redline=OPUS_REDLINE, manifest=manifest)
    out = tmp_path / "out"
    write_leaderboard(consolidate(runs), out)
    data = json.loads((out / "leaderboard.json").read_text(encoding="utf-8"))
    assert [r["model"] for r in data] == ["b-high-useful", "a-high-raw"]
