"""Build-time gate INVOCATION pins (robustness — 3-auditor review gap).

`build()` wires up several build-time gates, but only `f1_spotcheck` had a test
proving `build()` actually INVOKES it (see test_f1_gate.py). The auditor verified
that deleting the wiring for the *other* gates left every existing gate test green
— so a silently-removed or disabled gate would NOT fail CI.

These tests close that hole. Each one monkeypatches a gate in `build_mod` with a
spy that records every call (delegating to the real gate so the real build still
passes over the real backbone), runs `build(...)`, and asserts the spy was invoked
the exact number of times build()'s control flow demands. Removing a gate's call
site in build() therefore fails CI.

Expected call counts (PUBLIC_SPLITS == ("lite", "main", "diamond"), 3 public
splits, plus the private mirror):
  * safe_completion_spotcheck    — per public split + private        = 4
  * trap_mechanism_spotcheck     — per public split + private        = 4
  * coverage_spotcheck           — per public split + private        = 4
  * must_not_reachable_spotcheck — per public split (if/try branch,
                                    exactly once each) + private      = 4
  * consent_over_limit_count     — once, on by_split["main"] only     = 1
"""

from __future__ import annotations

from spar.dataset import build as build_mod
from spar.dataset.build import PUBLIC_SPLITS, build


def test_build_invokes_safe_completion_gate_on_every_split(tmp_path, monkeypatch):
    seen: list[int] = []
    real = build_mod.safe_completion_spotcheck

    def _spy(samples, *, enforce=True, split="main"):
        seen.append(len(samples))
        return real(samples, enforce=enforce, split=split)

    monkeypatch.setattr(build_mod, "safe_completion_spotcheck", _spy)
    build(public_dir=tmp_path / "pub", private_dir=tmp_path / "priv",
          build_seed=1, spar_version="1.0.0")
    # one call per public split + one for private.
    assert len(seen) == len(PUBLIC_SPLITS) + 1 == 4


def test_build_invokes_trap_mechanism_gate_on_every_split(tmp_path, monkeypatch):
    seen: list[int] = []
    real = build_mod.trap_mechanism_spotcheck

    def _spy(samples, *, enforce=False, split="main", **kw):
        seen.append(len(samples))
        return real(samples, enforce=enforce, split=split, **kw)

    monkeypatch.setattr(build_mod, "trap_mechanism_spotcheck", _spy)
    build(public_dir=tmp_path / "pub", private_dir=tmp_path / "priv",
          build_seed=1, spar_version="1.0.0")
    assert len(seen) == len(PUBLIC_SPLITS) + 1 == 4


def test_build_invokes_coverage_gate_on_every_split(tmp_path, monkeypatch):
    seen: list[int] = []
    real = build_mod.coverage_spotcheck

    def _spy(samples, *, split, enforce):
        seen.append(len(samples))
        return real(samples, split=split, enforce=enforce)

    monkeypatch.setattr(build_mod, "coverage_spotcheck", _spy)
    build(public_dir=tmp_path / "pub", private_dir=tmp_path / "priv",
          build_seed=1, spar_version="1.0.0")
    assert len(seen) == len(PUBLIC_SPLITS) + 1 == 4


def test_build_invokes_must_not_reachable_gate_on_every_split(tmp_path, monkeypatch):
    seen: list[int] = []
    real = build_mod.must_not_reachable_spotcheck

    def _spy(samples):
        seen.append(len(samples))
        return real(samples)

    monkeypatch.setattr(build_mod, "must_not_reachable_spotcheck", _spy)
    build(public_dir=tmp_path / "pub", private_dir=tmp_path / "priv",
          build_seed=1, spar_version="1.0.0")
    # called exactly once per public split (whichever of the if/try branch runs)
    # plus once for private.
    assert len(seen) == len(PUBLIC_SPLITS) + 1 == 4


def test_build_invokes_consent_over_limit_floor_on_main(tmp_path, monkeypatch):
    seen: list[int] = []
    real = build_mod.consent_over_limit_count

    def _spy(samples):
        seen.append(len(samples))
        return real(samples)

    monkeypatch.setattr(build_mod, "consent_over_limit_count", _spy)
    build(public_dir=tmp_path / "pub", private_dir=tmp_path / "priv",
          build_seed=1, spar_version="1.0.0")
    # the consent over-limit/scope-wall floor is asserted once, on by_split["main"].
    assert len(seen) == 1
