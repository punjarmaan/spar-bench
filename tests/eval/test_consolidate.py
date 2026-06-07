"""Consolidation behavior tests (offline, golden-fixture driven)."""

from __future__ import annotations

import json
from pathlib import Path

from spar.eval.consolidate import (
    AXES_ORDER,
    INTENT_SPECS_ORDER,
    LeaderboardEntry,
    consolidate,
    trust_useful_ci95,
    write_leaderboard,
)


def _make_entry(**overrides: object) -> LeaderboardEntry:
    base: dict[str, object] = {
        "model": "opus-frontier", "cls": "frontier",
        "trust_score": 0.71, "trust_score_ci95": (0.67, 0.75),
        "trust_score_raw": 0.71, "trust_score_objective": 0.70, "unsafe_completion_rate": 0.03,
        "false_refusal_rate": 0.09, "pass_1": 0.74, "pass_4": 0.62,
        "axes": {a: 0.7 for a in AXES_ORDER},
        "by_intent_spec": {i: 0.6 for i in INTENT_SPECS_ORDER},
        "n_main": 8, "n_redline": 6, "scored_fraction": 1.0, "status": "verified",
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
               "trust_score_ci95": (0.18, 0.24), "trust_score_raw": 0.21,
               "trust_score_objective": 0.20,
               "unsafe_completion_rate": 0.40, "false_refusal_rate": 0.05, "pass_1": 0.30,
               "pass_4": 0.10, "axes": {a: 0.2 for a in AXES_ORDER},
               "by_intent_spec": {i: 0.15 for i in INTENT_SPECS_ORDER},
               "n_main": 5, "n_redline": 6, "scored_fraction": 0.90, "status": "partial",
               "cost_usd": 0.15, "provenance": "public_self_run",
               "model_version_pin": "meta-llama/llama-3-70b@2026-xx",
               "scaffold_version": "1.0.0", "spar_version": "0.1.0",
               "dataset_canary": "spar:abc", "run_date": "2026-05-29"}
    entry = LeaderboardEntry.model_validate(payload)
    assert entry.cls == "open"
    assert entry.status == "partial"


def _ci_row(axis: str, score: float, *, is_trap: bool = False, cat_app: bool = False,
            cat_cls: str | None = None, completed: bool = True, unsafe: bool = False) -> dict:
    return {"axis": axis, "score": score, "is_trap": is_trap,
            "catastrophic_applicable": cat_app, "catastrophic_class": cat_cls,
            "final_state": "CLOSED" if completed else "ABORTED",
            "unsafe_completion": unsafe}


def test_trust_useful_ci95_is_ordered_and_brackets_the_statistic() -> None:
    # All-completed non-traps on one axis, no gates: the statistic reduces to the mean.
    scores = [0.7, 0.8, 0.6, 0.9, 0.7, 0.65, 0.75, 0.72]
    rows = [_ci_row("routing", s) for s in scores]
    lo, hi = trust_useful_ci95(rows)
    assert 0.0 <= lo <= hi <= 1.0
    assert lo <= sum(scores) / len(scores) <= hi   # the statistic lies inside its own 95% CI


def test_trust_useful_ci95_is_deterministic_across_two_calls() -> None:
    rows = [_ci_row("routing", s) for s in [0.7, 0.8, 0.6, 0.9, 0.7, 0.65, 0.75, 0.72]]
    assert trust_useful_ci95(rows) == trust_useful_ci95(rows)   # byte-identical (seeded RNG)


def test_trust_useful_ci95_is_gated_by_refusals() -> None:
    # A total-refusal model's trust_useful is robustly 0 — the CI must be (0, 0), never a
    # wide band (the old per-sample-score bootstrap measured an ungated statistic).
    rows = [_ci_row("routing", 0.0, completed=False) for _ in range(50)]
    assert trust_useful_ci95(rows) == (0.0, 0.0)


def test_trust_useful_ci95_floors_negative_competence() -> None:
    # Negative per-sample scores floor at 0 through the competence gate (max(0, comp)).
    rows = [_ci_row("routing", -0.5) for _ in range(10)]
    assert trust_useful_ci95(rows) == (0.0, 0.0)


def test_trust_useful_ci95_degenerate_single_row() -> None:
    lo, hi = trust_useful_ci95([_ci_row("routing", 0.5)])
    assert lo == hi == 0.5            # every resample is the same single row


def test_trust_useful_ci95_empty_is_zero_zero() -> None:
    assert trust_useful_ci95([]) == (0.0, 0.0)


def test_consolidate_single_model_pulls_competence_from_main(single_runs_dir: Path) -> None:
    entries = consolidate(single_runs_dir)
    assert len(entries) == 1
    e = entries[0]
    assert e.model == "opus-frontier"
    assert e.cls == "frontier"
    assert e.trust_score == 0.71                  # summary.trust_score from main
    assert e.trust_score_objective == 0.70        # summary.trust_score_objective from main
    assert e.unsafe_completion_rate == 0.03
    assert e.false_refusal_rate == 0.09
    assert e.pass_1 == 0.74                        # main pass^1
    assert e.pass_4 == 0.62                        # redline pass^4
    assert e.n_main == 8                           # main summary.n_samples
    assert e.n_redline == 6                        # redline summary.n_samples


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
    main = json.loads((single_runs_dir / "opus-frontier" / "main.results.json").read_text())
    assert e.trust_score_ci95 == trust_useful_ci95(main["per_sample"])


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
    # Strip provenance from the manifest -> default to public_self_run.
    manifest_path = single_runs_dir / "opus-frontier" / "run_manifest.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    del data["provenance"]
    manifest_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    e = consolidate(single_runs_dir)[0]
    assert e.provenance == "public_self_run"


def test_consolidate_returns_all_models(runs_dir: Path) -> None:
    entries = consolidate(runs_dir)
    assert {e.model for e in entries} == {"opus-frontier", "llama-open"}
    # consolidate orders by directory name (deterministic input order), NOT by trust.
    assert [e.model for e in entries] == ["llama-open", "opus-frontier"]


def test_consolidate_ignores_dirs_without_main_results(runs_dir: Path) -> None:
    (runs_dir / "trajectories-junk").mkdir()        # a stray dir with no main.results.json
    entries = consolidate(runs_dir)
    assert {e.model for e in entries} == {"opus-frontier", "llama-open"}


def test_write_leaderboard_json_sort_order(runs_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    write_leaderboard(consolidate(runs_dir), out)
    data = json.loads((out / "leaderboard.json").read_text(encoding="utf-8"))
    # private_verified above public_self_run, then trust_score desc.
    assert [row["model"] for row in data] == ["opus-frontier", "llama-open"]
    assert data[0]["provenance"] == "private_verified"
    assert data[0]["class"] == "frontier"        # serialized via alias
    assert data[0]["trust_score_ci95"] == list(data[0]["trust_score_ci95"])  # JSON list


def test_write_leaderboard_json_verified_above_unverified() -> None:
    # A higher-trust public_self_run still sorts BELOW any private_verified (provenance first).
    from tests.eval.conftest import (
        OPUS_REDLINE, OPUS_MAIN, _manifest, _write_model)
    import tempfile
    tmp = Path(tempfile.mkdtemp())
    runs = tmp / "runs"
    # high-trust public model
    hi_pub = _manifest(model="hi-pub", cls="open", provenance="public_self_run",
                       cost_usd=1.0, version_pin="x@1", canary="spar:abc")
    _write_model(runs / "hi-pub", main=OPUS_MAIN, redline=OPUS_REDLINE, manifest=hi_pub)
    # lower-trust private-verified model
    lo_priv_main = json.loads(json.dumps(OPUS_MAIN))
    lo_priv_main["summary"]["trust_score"] = 0.10
    lo_priv = _manifest(model="lo-priv", cls="frontier", provenance="private_verified",
                        cost_usd=2.0, version_pin="y@1", canary="spar:abc")
    _write_model(runs / "lo-priv", main=lo_priv_main, redline=OPUS_REDLINE, manifest=lo_priv)
    out = tmp / "out"
    write_leaderboard(consolidate(runs), out)
    data = json.loads((out / "leaderboard.json").read_text(encoding="utf-8"))
    assert [r["model"] for r in data] == ["lo-priv", "hi-pub"]   # verified first despite lower trust


def test_write_leaderboard_json_has_trailing_newline(runs_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    write_leaderboard(consolidate(runs_dir), out)
    assert (out / "leaderboard.json").read_text(encoding="utf-8").endswith("\n")


def test_write_leaderboard_csv_columns(runs_dir: Path, tmp_path: Path) -> None:
    import csv
    out = tmp_path / "out"
    write_leaderboard(consolidate(runs_dir), out)
    with (out / "leaderboard.csv").open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        header = reader.fieldnames
        rows = list(reader)
    assert header is not None
    # scalar columns present
    for col in ("model", "class", "trust_score", "trust_ci95_lo", "trust_ci95_hi",
                "trust_score_objective", "unsafe_completion_rate", "false_refusal_rate",
                "pass_1", "pass_4", "n_main", "n_redline", "scored_fraction", "status",
                "cost_usd", "provenance", "model_version_pin", "scaffold_version",
                "spar_version", "dataset_canary", "run_date"):
        assert col in header, col
    # axes expanded: one column per axis with an `axis_` prefix
    for axis in AXES_ORDER:
        assert f"axis_{axis}" in header
    # by_intent_spec expanded with an `intent_` prefix
    for intent in INTENT_SPECS_ORDER:
        assert f"intent_{intent}" in header
    # rows are in the published sort order, values are flattened correctly
    assert [r["model"] for r in rows] == ["opus-frontier", "llama-open"]
    opus = rows[0]
    assert opus["class"] == "frontier"
    assert float(opus["axis_routing"]) == 0.81
    assert float(opus["intent_semantic"]) == 0.61
    assert float(opus["trust_ci95_lo"]) <= float(opus["trust_ci95_hi"])


def test_write_leaderboard_csv_has_unix_line_endings(runs_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    write_leaderboard(consolidate(runs_dir), out)
    raw = (out / "leaderboard.csv").read_bytes()
    assert b"\r\n" not in raw           # deterministic \n terminator only


def test_leaderboard_md_consolidated_table(runs_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    write_leaderboard(consolidate(runs_dir), out)
    md = (out / "LEADERBOARD.md").read_text(encoding="utf-8")
    # Consolidated table header.
    assert "| Rank | Model | Class | Trust (±95%) | Trust (raw) | Unsafe-completion | pass^4 | Cost | Provenance |" in md
    # rows in published order; rank 1 is the verified frontier model.
    assert "| 1 | opus-frontier | frontier |" in md
    assert "| 2 | llama-open | open |" in md
    # Trust with CI rendered, cost as USD, provenance badge text present.
    assert "0.71" in md and "private_verified" in md and "public_self_run" in md
    assert "$4.10" in md


def test_leaderboard_md_per_axis_table(runs_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    write_leaderboard(consolidate(runs_dir), out)
    md = (out / "LEADERBOARD.md").read_text(encoding="utf-8")
    assert "## Per-axis breakdown" in md
    # Model x 7 axes + objective column header.
    for axis in AXES_ORDER:
        assert axis in md
    assert "objective" in md.lower()
    assert "0.81" in md          # opus routing axis cell


def test_leaderboard_md_framing_and_ci_note(runs_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    write_leaderboard(consolidate(runs_dir), out)
    md = (out / "LEADERBOARD.md").read_text(encoding="utf-8")
    # Framing prose — substring match on the key clause.
    assert "Trust = safe **and** competent, gated by overspend" in md
    # Per-stage sampling config + run date disclosed.
    assert "temperature" in md.lower()
    assert "2026-05-29" in md
    # Statistical-honesty note.
    assert "differences within overlapping CIs are not significant" in md


def test_leaderboard_md_has_trailing_newline(runs_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    write_leaderboard(consolidate(runs_dir), out)
    assert (out / "LEADERBOARD.md").read_text(encoding="utf-8").endswith("\n")


def test_leaderboard_manifest_union(runs_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    write_leaderboard(consolidate(runs_dir), out)
    manifest = json.loads((out / "leaderboard_manifest.json").read_text(encoding="utf-8"))
    assert manifest["spar_version"] == "0.1.0"
    assert manifest["scaffold_version"] == "1.0.0"
    assert sorted(manifest["dataset_canaries"]) == \
        ["spar:00000000-0000-0000-0000-000000000000"]
    assert manifest["bootstrap_seed"] == 12345          # documents the CI seed
    assert manifest["publishability_floor"] == 0.98
    # per-model pinned inputs union: model -> version pin
    assert manifest["model_version_pins"] == {
        "llama-open": "meta-llama/llama-3-70b@2026-xx",
        "opus-frontier": "anthropic/claude-opus-4@2026-xx"}
    assert sorted(manifest["run_dates"]) == ["2026-05-29"]


def test_leaderboard_manifest_has_trailing_newline(runs_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    write_leaderboard(consolidate(runs_dir), out)
    assert (out / "leaderboard_manifest.json").read_text(
        encoding="utf-8").endswith("\n")


def test_full_rerun_is_byte_identical(runs_dir: Path, tmp_path: Path) -> None:
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    write_leaderboard(consolidate(runs_dir), out_a)
    write_leaderboard(consolidate(runs_dir), out_b)
    for name in ("leaderboard.json", "leaderboard.csv", "LEADERBOARD.md",
                 "leaderboard_manifest.json"):
        assert (out_a / name).read_bytes() == (out_b / name).read_bytes(), name


def test_overwrite_in_place_is_idempotent(runs_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    write_leaderboard(consolidate(runs_dir), out)
    first = {n: (out / n).read_bytes() for n in
             ("leaderboard.json", "leaderboard.csv", "LEADERBOARD.md",
              "leaderboard_manifest.json")}
    write_leaderboard(consolidate(runs_dir), out)     # overwrite same dir
    for name, blob in first.items():
        assert (out / name).read_bytes() == blob, name


def test_consolidate_on_real_em2_output_tolerates_sparse_axes(tmp_path: Path) -> None:
    """Integration: real evaluate_model output (sparse per_axis) consolidates without crashing
    and fills every canonical axis/intent key (missing -> 0.0)."""
    import json as _json

    from spar.eval.cache import CompletionCache
    from spar.eval.models import ModelConfig
    from spar.eval.orchestrator import evaluate_model
    from spar.eval.profile import Profile, StagePlan, StageSampling
    from spar.harness.model_grader import StubModelGrader
    from spar.harness.user_sim import ScriptedUserSim, UserResponse

    def fake(*, model, messages, **s):  # offline completion_fn: always a valid abort action
        content = _json.dumps({"tool": "abort", "args": {"reason": "x"}})
        return type("R", (), {
            "choices": [type("C", (), {"message": type("M", (), {"content": content})()})()],
            "usage": type("U", (), {"prompt_tokens": 100, "completion_tokens": 20})(),
            "_hidden_params": {"response_cost": 0.002},
        })()

    runs = tmp_path / "runs"
    model = ModelConfig(id="realmodel", route="fake/r", price_in_per_mtok=1.0,
                        price_out_per_mtok=2.0, version_pin="fake/r@2026", **{"class": "open"})
    profile = Profile(
        competence=StageSampling(temperature=0.0, top_p=1.0, max_tokens=64, seed=7),
        reliability=StageSampling(temperature=0.7, top_p=1.0, max_tokens=64, seed=7),
        plan=[StagePlan(split="main", k=1, stage="competence", published=True),
              StagePlan(split="redline", k=4, stage="reliability", published=True)],
    )
    evaluate_model(
        model, profile, out_dir=runs, cache=CompletionCache(tmp_path / "c"),
        budget_usd=None, concurrency=1,
        responder=ScriptedUserSim(UserResponse(decision="deny")), grader=StubModelGrader(),
        completion_fn=fake, retries=1, sleep=lambda _s: None,
    )
    entries = consolidate(runs)
    assert len(entries) == 1
    e = entries[0]
    assert e.model == "realmodel"
    assert e.model_version_pin == "fake/r@2026"           # real manifest key
    assert list(e.axes.keys()) == AXES_ORDER              # all 7 present even if main was sparse
    assert list(e.by_intent_spec.keys()) == INTENT_SPECS_ORDER
    assert all(isinstance(v, float) for v in e.axes.values())
    # the leaderboard writer must also not crash on this entry
    write_leaderboard(entries, tmp_path / "board")
    assert (tmp_path / "board" / "leaderboard.json").exists()


def test_cli_leaderboard_writes_all_artifacts(runs_dir: Path, tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from spar.harness.run_eval import app

    out = tmp_path / "board"
    result = CliRunner().invoke(
        app, ["leaderboard", "--runs", str(runs_dir), "--out-dir", str(out)])
    assert result.exit_code == 0, result.output
    for name in ("leaderboard.json", "leaderboard.csv", "LEADERBOARD.md",
                 "leaderboard_manifest.json"):
        assert (out / name).exists(), name
    data = json.loads((out / "leaderboard.json").read_text(encoding="utf-8"))
    assert [r["model"] for r in data] == ["opus-frontier", "llama-open"]
    assert "wrote" in result.output.lower()
