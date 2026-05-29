"""ModelAgent (spec §5.1): the agent-under-test driven by a uniform JSON-action protocol.

One code path for every model (native tool-calling rejected — not uniform across OpenRouter's open
models, would confound the comparison). The system prompt is policy + mandate + a tool catalog
derived from the `contract` pydantic models (so it never drifts) + the JSON output contract. Each
`act` appends the rendered observation to an instance transcript, calls the INJECTED `completion_fn`,
json.loads -> parse_action -> Action; one bounded reformat retry; else Abort(malformed_action).

litellm is NOT imported here — the completion_fn is injected (the user_sim / model_grader pattern).
Cost is float USD (Decimal stays reserved for simulator money).
"""

from __future__ import annotations

import json

from spar.simulator.contract import (
    Abort,
    Capture,
    ComputeTax,
    HandleChallenge,
    ModifyCart,
    RequestUserConfirmation,
    Retry,
    SelectRoute,
    SubmitAuthorization,
)

# Bump on ANY change to the prompt template, tool catalog, or parser (spec standard 1).
SCAFFOLD_VERSION = "1.0.0"

# The 9 tool models, keyed by their `tool` Literal — the real action space (contract.py).
_TOOL_MODELS = (
    SelectRoute, ComputeTax, SubmitAuthorization, HandleChallenge, Retry,
    ModifyCart, RequestUserConfirmation, Capture, Abort,
)


def _tool_name(tool_schema: dict) -> str:
    """Extract the `tool` Literal value from its JSON-schema fragment.

    A single-value `Literal[...]` may serialize as `{"const": "..."}` or `{"enum": ["..."]}`
    depending on the pydantic version; accept either so the catalog never breaks on an upgrade.
    """
    if "const" in tool_schema:
        return tool_schema["const"]
    return tool_schema["enum"][0]


def tool_catalog() -> str:
    """A stable JSON string mapping each tool name -> its arg JSON-schema.

    Derived from the contract pydantic models via `model_json_schema()`, so the catalog can never
    drift from the real action space. The `tool` discriminator field is stripped from `properties`
    and `required` (the agent supplies it as the top-level "tool" key, not inside args).
    """
    catalog: dict[str, dict] = {}
    for model in _TOOL_MODELS:
        schema = model.model_json_schema()
        props = {k: v for k, v in schema.get("properties", {}).items() if k != "tool"}
        # The Literal["..."] on `tool` is the dict key.
        name = _tool_name(schema["properties"]["tool"])
        required = [r for r in schema.get("required", []) if r != "tool"]
        catalog[name] = {"properties": props, "required": required}
    return json.dumps(catalog, sort_keys=True)
