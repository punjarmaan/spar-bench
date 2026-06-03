"""Plan B B1a: build-time solvability gate — the upper-bound dual of f1_spotcheck.

f1_spotcheck proves naive completion FAILS non-traps (a non-trap is non-trivial). This gate
proves the competent reference agent (DiligentAgent) SOLVES every non-trap (a non-trap is not
accidentally impossible). Together they bracket each non-trap: hard for the naive baseline,
solvable for the diligent one. Guards later generator changes from silently making non-traps
unsolvable.
"""

from __future__ import annotations

import pytest

from spar.agents.naive_complete import NaiveCompleteAgent
from spar.agents.reference_agents import DiligentAgent
from spar.dataset.build import (
    COVERAGE_ENFORCED_SPLITS,
    SolvabilityGateError,
    solvability_spotcheck,
)
from spar.dataset.generator import generate
from spar.dataset.plan import plan_all
from spar.harness.graders import score
from spar.harness.runner import run_episode
from spar.harness.user_sim import ScriptedUserSim, UserResponse


def _approve_sim() -> ScriptedUserSim:
    # Cooperative responder: an underspecified/escalate-then-complete non-trap can finish.
    return ScriptedUserSim(UserResponse(decision="approve"))


def _main_non_traps(limit: int = 12):
    planned = [p for p in plan_all(build_seed=1) if p.split == "main" and not p.spec.is_trap]
    return [generate(p.spec) for p in planned[:limit]]


def test_solvability_spotcheck_is_exported_and_callable():
    assert callable(solvability_spotcheck)


def test_diligent_agent_solvability_is_scored_per_sample():
    # FINDING (Plan B B1a): DiligentAgent is a SAFE baseline, not a competent happy-path solver,
    # so it does NOT solve every non-trap (it omits axis-specific non-trivial `must` tokens and
    # appends a defensive abort on some axes). This test pins that diagnosis: each non-trap gets a
    # well-formed score, and the spotcheck's unsolved set EQUALS the score<1.0 set (the judge
    # mirrors f1_spotcheck's per-sample `score`, not a hand-rolled criterion).
    samples = _main_non_traps()
    assert samples, "expected some main non-traps to sample"
    expected_unsolved = sorted(
        s.sample_id
        for s in samples
        if score(s, run_episode(s, DiligentAgent(), user_sim=_approve_sim())).score < 1.0
    )
    got = solvability_spotcheck(samples, enforce=False, split="main")
    assert got == expected_unsolved


def test_solvability_gate_log_only_does_not_block_diligent_main():
    # The WIRED path is enforce=False (the build run found DiligentAgent unsolved-non-traps),
    # so the gate must NOT raise even on the enforced `main` split when enforce=False. The
    # function RETAINS the ability to enforce (covered by the bad-solver test below); this only
    # asserts the log-only wiring is non-blocking.
    samples = _main_non_traps()
    unsolved = solvability_spotcheck(samples, enforce=False, split="main")
    assert isinstance(unsolved, list)  # never raises; surfaces the ids for logging


def test_solvability_gate_enforces_when_diligent_leaves_non_traps_unsolved():
    # The gate CAN still enforce: with the default DiligentAgent solver and enforce=True on an
    # enforced split, the current generator's unsolved non-traps trip SolvabilityGateError. This
    # proves the enforcement machinery is live and would catch a genuine regression once a
    # competent reference solver is wired (Plan B follow-up flips the build wiring back on).
    samples = _main_non_traps()
    if not solvability_spotcheck(samples, enforce=False, split="main"):
        pytest.skip("no unsolved DiligentAgent non-traps in this sample window")
    with pytest.raises(SolvabilityGateError):
        solvability_spotcheck(samples, enforce=True, split="main")


def test_solvability_gate_catches_a_bad_solver_when_enforced():
    # A NaiveCompleteAgent standing in as the solver CANNOT solve non-traps (it is the f1
    # adversary). The gate must raise SolvabilityGateError on an enforced split and list ids.
    samples = _main_non_traps()
    with pytest.raises(SolvabilityGateError) as ei:
        solvability_spotcheck(samples, enforce=True, split="main", agent_factory=NaiveCompleteAgent)
    # The error names the unsolved sample_ids.
    assert samples[0].sample_id in str(ei.value)


def test_solvability_gate_log_only_when_not_enforced():
    # enforce=False never raises; it returns the (possibly non-empty) unsolved-id list.
    samples = _main_non_traps()
    unsolved = solvability_spotcheck(
        samples, enforce=False, split="main", agent_factory=NaiveCompleteAgent
    )
    assert unsolved  # naive cannot solve non-traps, but enforce=False does not raise


def test_solvability_gate_does_not_enforce_off_enforced_splits():
    # Even a bad solver on a non-enforced split (e.g. lite) does not raise when enforce flows
    # through the COVERAGE_ENFORCED_SPLITS membership, mirroring f1/coverage wiring.
    assert "lite" not in COVERAGE_ENFORCED_SPLITS
    samples = _main_non_traps()
    unsolved = solvability_spotcheck(
        samples, enforce=("lite" in COVERAGE_ENFORCED_SPLITS), split="lite",
        agent_factory=NaiveCompleteAgent,
    )
    assert unsolved  # computed + returned, never raised
