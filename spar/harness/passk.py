"""pass^k reliability (module 40 §3.3, F7).

Unbiased all-pass estimator C(c,k)/C(n,k) — the hypergeometric probability that all k of k
trials drawn without replacement from n trials (c solved) are solved. The biased plug-in
(c/n)^k is NOT used. "Solved in a trial" = sample_score >= pass_threshold.
"""

from __future__ import annotations

from math import comb

PASS_THRESHOLD_BINARY: float = 1.0
PASS_THRESHOLD_ROUTING: float = 0.99


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
