"""pass^k reliability (module 40 §3.3, F7).

Unbiased all-pass estimator C(c,k)/C(n,k) — the hypergeometric probability that all k of k
trials drawn without replacement from n trials (c solved) are solved. The biased plug-in
(c/n)^k is NOT used. "Solved in a trial" = sample_score >= pass_threshold.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import comb

from spar.agents.base import Agent
from spar.harness.graders import score
from spar.harness.runner import run_episode
from spar.simulator.enums import Axis
from spar.simulator.schemas import Sample

PASS_THRESHOLD_BINARY: float = 1.0
PASS_THRESHOLD_ROUTING: float = 0.99

AgentFactory = Callable[[], Agent]


def passk_estimate(n: int, c: int, k: int) -> float:
    """Unbiased pass^k = C(c,k)/C(n,k); 0 when c < k; pass^1 = c/n. Never (c/n)^k."""
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if n < k:
        raise ValueError(f"n must be >= k, got n={n}, k={k}")
    if not (0 <= c <= n):
        raise ValueError(f"c must be in [0, n], got c={c}, n={n}")
    if c < k:
        return 0.0
    return comb(c, k) / comb(n, k)


def is_solved(score: float, *, binary: bool) -> bool:
    """A trial is solved iff its score meets the pass threshold (1.0 binary / 0.99 routing)."""
    threshold = PASS_THRESHOLD_BINARY if binary else PASS_THRESHOLD_ROUTING
    return score >= threshold


@dataclass
class TrialResult:
    n: int          # trials run
    c: int          # trials solved (score >= threshold)
    passk: float    # C(c,k)/C(n,k)


def run_trials(sample: Sample, agent_factory: AgentFactory, *, k: int) -> TrialResult:
    """Run a live importable agent k times, re-seeding the world per trial (F7), estimate pass^k.

    Each trial builds a FRESH agent (a stochastic LLM agent is re-run, never replayed) and
    re-seeds the world via trial_index, which the World feeds to derive_seed(sample.seed, ...).
    """
    binary = sample.axis is not Axis.ROUTING
    solved = 0
    for trial_index in range(k):
        trace = run_episode(sample, agent_factory(), trial_index=trial_index)
        if is_solved(score(sample, trace).score, binary=binary):
            solved += 1
    return TrialResult(n=k, c=solved, passk=passk_estimate(k, solved, k))
