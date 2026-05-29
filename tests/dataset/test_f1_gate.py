"""F1 build-time gate (CODE-REVIEW H2).

The F1 generation invariant (module 30 §3) is enforced in code by `f1_spotcheck`, but
`build()` never invoked it — so a regression introducing a naive-completable non-trap into a
shipped split would ship silently. These tests pin: (a) the spot-check actually RAISES on a
gameable non-trap set, and (b) `build()` runs it on every shipped split.
"""

from __future__ import annotations

import pytest

from spar.dataset import build as build_mod
from spar.dataset.build import build, f1_spotcheck
from spar.dataset.gold_backbone import load_gold_backbone


def test_f1_spotcheck_raises_on_gameable_nontraps():
    # The full hand-authored backbone deliberately includes trivially-completable easy gold
    # (for gold_replay) -> naive completion solves >10% of its non-traps, so the 0.9 floor
    # must RAISE. This proves the gate has teeth.
    with pytest.raises(AssertionError, match="F1 violation"):
        f1_spotcheck(load_gold_backbone(), floor=0.9)


def test_build_invokes_f1_gate_on_every_shipped_split(tmp_path, monkeypatch):
    seen: list[int] = []  # record the size of each split the gate is run against

    real = build_mod.f1_spotcheck

    def _spy(samples, *, floor=0.9):
        seen.append(len(samples))
        return real(samples, floor=floor)

    monkeypatch.setattr(build_mod, "f1_spotcheck", _spy)
    build(public_dir=tmp_path / "pub", private_dir=tmp_path / "priv",
          build_seed=1, spar_version="1.0.0")
    # gate must run on lite + main + diamond + private (4 shipped graded sets).
    assert len(seen) == 4


def test_build_succeeds_because_shipped_splits_pass_f1(tmp_path):
    # Real build over real data must NOT raise: every shipped split clears the F1 floor
    # (procedural generator keeps non-traps non-trivial; diamond is all traps).
    build(public_dir=tmp_path / "pub", private_dir=tmp_path / "priv",
          build_seed=7, spar_version="1.0.0")
    assert (tmp_path / "pub" / "main.jsonl").exists()
