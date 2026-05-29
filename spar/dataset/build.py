"""Build orchestrator (module 30 §5; PLANS-REVIEW M7; F13/F14/F17).

Emits PUBLIC Lite/Main/Diamond through `public_view` (the F17 answer-leakage projection) into
`public_dir`, and the SERVER-SIDE/Private full-graded build into a SEPARATE `private_dir`.
Split membership comes from the SINGLE mechanism (`plan.plan_all`) for procedural splits and
the hand-authored `diamond_backbone` (F14) for Diamond — there is no second hash-bucketing
pass. One FRESH canary per build is stamped on every line and recorded in every manifest.
Published splits are FROZEN, hash-pinned artifacts (F13): eval loads them, never regenerates.
"""

from __future__ import annotations

import json
from pathlib import Path

from spar.agents.naive_complete import NaiveCompleteAgent
from spar.dataset.generator import generate
from spar.dataset.gold_backbone import diamond_backbone
from spar.dataset.manifest import build_manifest
from spar.dataset.plan import plan_all
from spar.dataset.projection import public_view
from spar.dataset.splits import apply_canary, make_canary
from spar.harness.graders import score
from spar.harness.runner import run_episode
from spar.simulator.schemas import Sample

PUBLIC_SPLITS: tuple[str, ...] = ("lite", "main", "diamond")
DIAMOND_CAP = 198
F1_FLOOR = 0.9   # H2: >=90% of each shipped split's non-traps must defeat naive completion


def _write_public(base: Path, split: str, samples: list[Sample], *,
                  build_seed: int, canary: str, spar_version: str) -> None:
    """Write a public split as projected JSONL + a manifest (computed on full Samples)."""
    samples = sorted(samples, key=lambda s: s.sample_id)
    lines = [json.dumps(public_view(s), sort_keys=True) for s in samples]
    body = "\n".join(lines)
    (base / f"{split}.jsonl").write_text(body + ("\n" if body else ""), encoding="utf-8")
    manifest = build_manifest(samples, split=split, build_seed=build_seed,
                              canary=canary, spar_version=spar_version)
    (base / f"{split}.manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def _write_private(base: Path, samples: list[Sample], *, build_seed: int,
                   canary: str, spar_version: str) -> None:
    """Write the FULL graded Private build (gold + hidden config), unprojected."""
    samples = sorted(samples, key=lambda s: s.sample_id)
    body = "\n".join(s.model_dump_json() for s in samples)
    (base / "private.jsonl").write_text(body + ("\n" if body else ""), encoding="utf-8")
    manifest = build_manifest(samples, split="private", build_seed=build_seed,
                              canary=canary, spar_version=spar_version)
    (base / "private.manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def build(*, public_dir: Path, private_dir: Path, build_seed: int,
          spar_version: str) -> None:
    """Build public Lite/Main/Diamond + a SEPARATE server-side Private build."""
    public_dir = Path(public_dir)
    private_dir = Path(private_dir)
    if public_dir.resolve() == private_dir.resolve():
        raise ValueError("public_dir and private_dir must differ (Private isolation)")
    public_dir.mkdir(parents=True, exist_ok=True)
    private_dir.mkdir(parents=True, exist_ok=True)
    canary = make_canary()  # fresh per build (PLANS-REVIEW M7)

    # Single mechanism: procedural samples already bound to lite/main.
    by_split: dict[str, list[Sample]] = {"lite": [], "main": []}
    private: list[Sample] = []
    for planned in plan_all(build_seed=build_seed):
        sample = apply_canary(generate(planned.spec), canary)
        by_split[planned.split].append(sample)
        # Private holds the full graded copy of every procedural sample too, so the
        # leaderboard can grade public submissions server-side against hidden gold.
        private.append(sample)

    # Diamond: hand-authored backbone only (F14), capped at 198.
    diamond = sorted(diamond_backbone(), key=lambda s: s.sample_id)[:DIAMOND_CAP]
    diamond = [apply_canary(s, canary) for s in diamond]
    by_split["diamond"] = diamond
    private.extend(diamond)

    # H2 (F1 gate): enforce the non-trivial-non-trap invariant on every SHIPPED graded split
    # before writing — a regression that let naive completion solve a non-trap would otherwise
    # ship silently. The gate runs on the built splits (procedural + diamond), NOT on the raw
    # easy-laden gold backbone (whose trivial gold is intentional, for gold_replay). Diamond is
    # all-traps -> the gate passes vacuously but is asserted so a future Diamond non-trap is caught.
    for split in PUBLIC_SPLITS:
        f1_spotcheck(by_split[split], floor=F1_FLOOR)
    f1_spotcheck(private, floor=F1_FLOOR)

    for split in PUBLIC_SPLITS:
        _write_public(public_dir, split, by_split[split], build_seed=build_seed,
                      canary=canary, spar_version=spar_version)
    _write_private(private_dir, private, build_seed=build_seed, canary=canary,
                   spar_version=spar_version)


def f1_spotcheck(samples: list[Sample], *, floor: float = 0.9, min_axis_n: int = 10) -> float:
    """Run NaiveCompleteAgent over non-trap samples; return the fraction it FAILS to complete.

    F1 (module 30 §3): a bare select->submit->capture must NOT complete a well-formed
    non-trap. The gate is enforced **per axis** (module 30 §3 says "per axis") for every axis
    with at least `min_axis_n` non-traps — an aggregate-only floor MASKS a single weak axis
    (e.g. a 35%-gameable post_purchase hides behind a 5% aggregate). Axes below `min_axis_n`
    (small splits like Lite) only contribute to the aggregate floor, to avoid 1-sample noise.
    Raises on any violation; otherwise returns the overall failed fraction.
    """
    non_traps = [s for s in samples if not s.is_trap]
    if not non_traps:
        return 1.0
    by_axis: dict[str, list[bool]] = {}
    for sample in non_traps:
        trace = run_episode(sample, NaiveCompleteAgent(), trial_index=0)
        failed = not score(sample, trace).outcome_correct
        by_axis.setdefault(sample.axis.value, []).append(failed)

    for axis, fails in sorted(by_axis.items()):
        if len(fails) >= min_axis_n:
            frac = sum(fails) / len(fails)
            if frac < floor:
                raise AssertionError(
                    f"F1 violation [axis={axis}]: naive completion solved "
                    f"{(1 - frac):.0%} of its {len(fails)} non-traps "
                    f"(floor requires it fail >= {floor:.0%})"
                )
    fraction = sum(sum(v) for v in by_axis.values()) / len(non_traps)
    if fraction < floor:
        raise AssertionError(
            f"F1 violation: naive completion solved {(1 - fraction):.0%} of non-traps "
            f"(floor requires it fail >= {floor:.0%})"
        )
    return fraction
