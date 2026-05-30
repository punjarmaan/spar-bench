"""EM3 — cross-model consolidation into the published leaderboard (spec §5.7-§5.10).

Pure merge + present layer: reads each model's main/diamond results.json + run_manifest.json
(produced upstream by spar/harness/report.py + the EM2 orchestrator) and emits
leaderboard.{json,csv,md} + leaderboard_manifest.json. It NEVER re-scores or calls a model.

Dependency direction inward: imports only stdlib + numpy + pydantic.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_serializer

# Canonical leaderboard column orders (mirror spar/simulator/enums.py Axis / IntentSpec values).
AXES_ORDER: list[str] = [
    "routing", "decline_recovery", "consent_mandate", "stale_state",
    "compliance_tax", "fraud_reactivity", "post_purchase",
]
INTENT_SPECS_ORDER: list[str] = ["explicit", "semantic", "underspecified"]

# The completeness gate (spec §5.5): a (model) result is `verified` only at/above this floor.
PUBLISHABILITY_FLOOR = 0.98

# Seed for the reproducible bootstrap. A fixed numpy.random.default_rng(TRUST_CI95_SEED) makes the
# CI byte-identical across runs (spec §5.8 — reproducibility from the seed, not from temp=0).
TRUST_CI95_SEED = 12345
TRUST_CI95_RESAMPLES = 2000


class LeaderboardEntry(BaseModel):
    """One published leaderboard row (spec §5.7).

    `cls` is the Python attribute; it serializes/validates under the JSON key `class` (a Python
    reserved word) via the alias. `populate_by_name=True` so callers may pass `cls=` directly.
    """

    model_config = ConfigDict(populate_by_name=True)

    model: str
    cls: Literal["frontier", "open"] = Field(alias="class")
    trust_score: float
    trust_score_ci95: tuple[float, float]
    trust_score_objective: float
    overspend_rate: float
    false_refusal_rate: float
    pass_1: float
    pass_4: float
    axes: dict[str, float]
    by_intent_spec: dict[str, float]
    n_main: int
    n_diamond: int
    scored_fraction: float
    status: Literal["verified", "partial"]
    cost_usd: float
    provenance: Literal["private_verified", "public_self_run"]
    model_version_pin: str
    scaffold_version: str
    spar_version: str
    dataset_canary: str
    run_date: str

    @field_serializer("trust_score_ci95")
    def _ser_ci95(self, value: tuple[float, float]) -> list[float]:
        """Serialize the CI tuple as a 2-element JSON list (deterministic, list-typed)."""
        return [value[0], value[1]]


def trust_ci95(per_sample_scores: list[float]) -> tuple[float, float]:
    """Deterministic 95% bootstrap CI over per-sample CLAMPED scores (spec §5.8).

    Each score is clamped to [0, 1]; we resample with replacement `TRUST_CI95_RESAMPLES` times
    using a fixed-seed numpy.random.default_rng(TRUST_CI95_SEED), take each resample's mean, and
    return the 2.5th / 97.5th percentiles. Empty -> (0.0, 0.0). A single score -> (s, s) since
    every resample is identical. The seed guarantees the same input yields byte-identical output.
    """
    if not per_sample_scores:
        return (0.0, 0.0)
    clamped = np.clip(np.asarray(per_sample_scores, dtype=np.float64), 0.0, 1.0)
    n = clamped.shape[0]
    rng = np.random.default_rng(TRUST_CI95_SEED)
    idx = rng.integers(0, n, size=(TRUST_CI95_RESAMPLES, n))
    means = clamped[idx].mean(axis=1)
    lo = float(np.percentile(means, 2.5))
    hi = float(np.percentile(means, 97.5))
    return (lo, hi)


def _read_json(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


def _build_entry(model_dir: Path) -> LeaderboardEntry:
    """Merge one model's main + diamond + run_manifest into a LeaderboardEntry."""
    main = _read_json(model_dir / "main.results.json")
    manifest = _read_json(model_dir / "run_manifest.json")

    msum = main["summary"]
    per_axis = main["per_axis"]
    by_intent = main["by_intent_spec"]

    # Robustness: a run that skipped the reliability stage may lack diamond.results.json.
    diamond_path = model_dir / "diamond.results.json"
    if diamond_path.exists():
        diamond = _read_json(diamond_path)
        dsum = diamond["summary"]
        pass_4 = dsum["pass_4"] or 0.0
        n_diamond_results = dsum["n_samples"]
    else:
        pass_4 = 0.0
        n_diamond_results = None

    per_sample_scores = [s["score"] for s in main["per_sample"]]

    # Completeness + version pin from the REAL EM2 manifest keys (splits.main.*, model_version_pin).
    splits = manifest.get("splits") or {}
    main_split = splits.get("main") or {}
    diamond_split = splits.get("diamond") or {}
    scored_fraction_val = main_split.get("scored_fraction")
    if scored_fraction_val is None:
        # Absent completeness cannot prove publishability -> partial. NEVER default 1.0 (spec gate
        # must be able to fire). (index manifest-contract reconciliation note.)
        scored_fraction = 0.0
        status: Literal["verified", "partial"] = "partial"
    else:
        scored_fraction = float(scored_fraction_val)
        status = main_split.get("status") or (
            "verified" if scored_fraction >= PUBLISHABILITY_FLOOR else "partial"
        )
    n_main = int(main_split.get("n", msum["n_samples"]))
    if n_diamond_results is not None:
        n_diamond = int(diamond_split.get("n", n_diamond_results))
    else:
        n_diamond = int(diamond_split.get("n", 0))

    # Provenance is a recorded fact (spec §5.9): default public_self_run unless the manifest
    # explicitly stamps private_verified (only the private grading server may do so).
    provenance = manifest.get("provenance", "public_self_run")

    return LeaderboardEntry(
        model=manifest["model"],
        cls=manifest["class"],
        trust_score=msum["trust_score"],
        trust_score_ci95=trust_ci95(per_sample_scores),
        trust_score_objective=msum["trust_score_objective"],
        overspend_rate=msum["overspend_rate"] or 0.0,
        false_refusal_rate=msum["false_refusal_rate"] or 0.0,
        pass_1=msum["pass_1"] or 0.0,
        pass_4=pass_4,
        axes={a: per_axis[a]["mean_score"] for a in AXES_ORDER},
        by_intent_spec={i: by_intent[i]["mean_score"] for i in INTENT_SPECS_ORDER},
        n_main=n_main,
        n_diamond=n_diamond,
        scored_fraction=scored_fraction,
        status=status,
        cost_usd=float(manifest.get("cost_usd", 0.0)),
        provenance=provenance,
        model_version_pin=manifest["model_version_pin"],
        scaffold_version=manifest["scaffold_version"],
        spar_version=manifest["spar_version"],
        dataset_canary=manifest["canary"],
        run_date=manifest["run_date"],
    )


def consolidate(runs_dir: Path) -> list[LeaderboardEntry]:
    """Read every runs/<model>/ dir (main + diamond + manifest) -> one LeaderboardEntry each.

    Returns entries in directory-name sorted order (deterministic input ordering); the published
    sort (provenance, then trust desc) is applied by write_leaderboard.
    """
    model_dirs = sorted(
        (d for d in runs_dir.iterdir()
         if d.is_dir() and (d / "main.results.json").exists()),
        key=lambda d: d.name,
    )
    return [_build_entry(d) for d in model_dirs]


# provenance sort rank: private_verified outranks public_self_run (spec §5.7/§5.9).
_PROVENANCE_RANK = {"private_verified": 0, "public_self_run": 1}


def _sorted_entries(entries: list[LeaderboardEntry]) -> list[LeaderboardEntry]:
    """Published order: private_verified before public_self_run, then trust_score descending.

    Ties broken by model name for total determinism.
    """
    return sorted(
        entries,
        key=lambda e: (_PROVENANCE_RANK[e.provenance], -e.trust_score, e.model),
    )


CSV_SCALAR_COLUMNS: list[str] = [
    "model", "class", "trust_score", "trust_ci95_lo", "trust_ci95_hi",
    "trust_score_objective", "overspend_rate", "false_refusal_rate", "pass_1", "pass_4",
    "n_main", "n_diamond", "scored_fraction", "status", "cost_usd", "provenance",
    "model_version_pin", "scaffold_version", "spar_version", "dataset_canary", "run_date",
]


def _csv_header() -> list[str]:
    return (CSV_SCALAR_COLUMNS
            + [f"axis_{a}" for a in AXES_ORDER]
            + [f"intent_{i}" for i in INTENT_SPECS_ORDER])


def _csv_row(e: LeaderboardEntry) -> dict[str, object]:
    lo, hi = e.trust_score_ci95
    row: dict[str, object] = {
        "model": e.model, "class": e.cls, "trust_score": e.trust_score,
        "trust_ci95_lo": lo, "trust_ci95_hi": hi,
        "trust_score_objective": e.trust_score_objective,
        "overspend_rate": e.overspend_rate, "false_refusal_rate": e.false_refusal_rate,
        "pass_1": e.pass_1, "pass_4": e.pass_4, "n_main": e.n_main, "n_diamond": e.n_diamond,
        "scored_fraction": e.scored_fraction, "status": e.status, "cost_usd": e.cost_usd,
        "provenance": e.provenance, "model_version_pin": e.model_version_pin,
        "scaffold_version": e.scaffold_version, "spar_version": e.spar_version,
        "dataset_canary": e.dataset_canary, "run_date": e.run_date,
    }
    for a in AXES_ORDER:
        row[f"axis_{a}"] = e.axes[a]
    for i in INTENT_SPECS_ORDER:
        row[f"intent_{i}"] = e.by_intent_spec[i]
    return row


# The framing prose + sampling disclosure + statistical-honesty note (spec §5.3/§5.7/§5.8).
_FRAMING = (
    "**Trust = safe **and** competent, gated by overspend.** A model that completes "
    "everything but over-spends on traps scores ~0.\n\n"
    "Sampling (identical for every model): competence stage (Main pass^1) temperature=0.0, "
    "top_p=1.0; reliability stage (Diamond pass^4) temperature=0.7, top_p=1.0. "
    "Reproducibility comes from the completion cache + pinned model snapshots, not from "
    "temperature=0.\n\n"
    "Statistical honesty: differences within overlapping CIs are not significant — rankings "
    "carry no false precision."
)


def _md_consolidated_table(ordered: list[LeaderboardEntry]) -> str:
    lines = [
        "## Consolidated leaderboard",
        "",
        "| Rank | Model | Class | Trust (±95%) | Overspend | pass^4 | Cost | Provenance |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for rank, e in enumerate(ordered, start=1):
        lo, hi = e.trust_score_ci95
        trust = f"{e.trust_score:.2f} ({lo:.2f}–{hi:.2f})"
        lines.append(
            f"| {rank} | {e.model} | {e.cls} | {trust} | {e.overspend_rate:.2f} | "
            f"{e.pass_4:.2f} | ${e.cost_usd:.2f} | {e.provenance} |"
        )
    return "\n".join(lines)


def _md_per_axis_table(ordered: list[LeaderboardEntry]) -> str:
    header = "| Model | " + " | ".join(AXES_ORDER) + " | objective |"
    sep = "| --- |" + " --- |" * (len(AXES_ORDER) + 1)
    lines = ["## Per-axis breakdown", "", header, sep]
    for e in ordered:
        cells = " | ".join(f"{e.axes[a]:.2f}" for a in AXES_ORDER)
        lines.append(f"| {e.model} | {cells} | {e.trust_score_objective:.2f} |")
    return "\n".join(lines)


def _render_markdown(ordered: list[LeaderboardEntry]) -> str:
    run_date = ordered[0].run_date if ordered else ""
    parts = [
        "# Spar Cross-Model Leaderboard",
        "",
        _FRAMING,
        "",
        f"_Run date: {run_date}_",
        "",
        _md_consolidated_table(ordered),
        "",
        _md_per_axis_table(ordered),
        "",
    ]
    return "\n".join(parts)


def _build_leaderboard_manifest(ordered: list[LeaderboardEntry]) -> dict[str, object]:
    """Top-level union of every pinned input needed to reconstruct any row (spec §5.10).

    Deterministic: sorted/unique collections, fixed scalar keys.
    """
    return {
        "spar_version": ordered[0].spar_version if ordered else "",
        "scaffold_version": ordered[0].scaffold_version if ordered else "",
        "bootstrap_seed": TRUST_CI95_SEED,
        "bootstrap_resamples": TRUST_CI95_RESAMPLES,
        "publishability_floor": PUBLISHABILITY_FLOOR,
        "dataset_canaries": sorted({e.dataset_canary for e in ordered}),
        "run_dates": sorted({e.run_date for e in ordered}),
        "model_version_pins": {e.model: e.model_version_pin for e in ordered},
        "models": sorted(e.model for e in ordered),
    }


def write_leaderboard(entries: list[LeaderboardEntry], out_dir: Path) -> None:
    """Emit leaderboard.json (canonical, sorted) + .csv + LEADERBOARD.md + manifest.

    Deterministic by construction: a fixed sort, JSON with sort_keys + a trailing newline, so a
    re-run over the same runs/ is byte-identical.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    ordered = _sorted_entries(entries)

    rows = [e.model_dump(by_alias=True) for e in ordered]
    # tuple CI -> JSON list happens automatically; sort_keys keeps key order stable.
    json_text = json.dumps(rows, indent=2, sort_keys=True) + "\n"
    (out_dir / "leaderboard.json").write_text(json_text, encoding="utf-8")

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=_csv_header(), lineterminator="\n")
    writer.writeheader()
    for e in ordered:
        writer.writerow(_csv_row(e))
    (out_dir / "leaderboard.csv").write_text(buf.getvalue(), encoding="utf-8")

    (out_dir / "LEADERBOARD.md").write_text(_render_markdown(ordered), encoding="utf-8")

    manifest_text = json.dumps(
        _build_leaderboard_manifest(ordered), indent=2, sort_keys=True) + "\n"
    (out_dir / "leaderboard_manifest.json").write_text(manifest_text, encoding="utf-8")
