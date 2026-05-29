"""Public-distribution projection (module 30 §5, REVIEW F17, index registry).

`public_view(sample)` emits ONLY the agent-visible surface — the redacted
Observation fields plus sample_id/axis/difficulty/policy_id/canary/seed — and
STRIPS the entire `gold` block (Spar's answer) and ALL hidden `world_config`
fields (`approval_prob`, `true_fee_bps`, `reliability`, `decline_plan` internals,
`market_context`). Public Lite/Main/Diamond are published through this; the full
graded Sample (gold + hidden config) lives only in the server-side/Private build.
A property test asserts the output contains no hidden/gold key.
"""

from __future__ import annotations

import json

from spar.simulator.schemas import Sample

# Hidden world_config keys that must never reach the public surface (the grader's
# ground truth / answer-bearing fields).
HIDDEN_WORLD_CONFIG_KEYS: tuple[str, ...] = (
    "approval_prob", "true_fee_bps", "reliability", "decline_plan",
    "market_context", "fraud_engine", "dispute", "settlement",
)

# Per-method fields the agent legitimately observes (module 20 §2; F2). Everything
# else on an acquirer (true_fee_bps, approval_prob, reliability) is hidden truth.
_VISIBLE_METHOD_KEYS: tuple[str, ...] = (
    "acquirer_id", "methods", "geos", "advertised_fee_bps", "observed_approval_band",
)


def public_view(sample: Sample) -> dict:  # type: ignore[type-arg]
    """The agent-visible projection of one Sample (F17). Strips gold + hidden config."""
    # Use JSON round-trip so Decimals/enums serialize exactly as published.
    raw = json.loads(sample.model_dump_json())
    wc = raw["world_config"]
    methods = [
        {k: acq[k] for k in _VISIBLE_METHOD_KEYS if k in acq}
        for acq in wc.get("acquirers", [])
    ]
    # Cart surface: subtotal + (post-tax) total only become visible as the episode
    # runs; at publish time we expose the static line-item shell the agent starts with.
    cart: dict[str, object] = {"line_items": [], "subtotal": None, "computed_tax": None,
                                "total": None, "alternatives": None}
    return {
        "sample_id": raw["sample_id"],
        "axis": raw["axis"],
        "difficulty": raw["difficulty"],
        "policy_id": raw["policy_id"],
        "canary": raw["canary"],
        "seed": raw["seed"],
        "mandate": raw["mandate"],   # incl. machine-enforced authority wall + budget_hint
        "cart": cart,
        "methods": methods,
    }
