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

from pydantic import BaseModel

from spar.eval.models import ModelConfig
from spar.eval.profile import StageSampling
from spar.harness.passk import AgentFactory
from spar.harness.user_sim import CompletionFn
from spar.simulator.contract import (
    Abort,
    Action,
    Capture,
    ComputeTax,
    HandleChallenge,
    ModifyCart,
    Observation,
    RequestUserConfirmation,
    Retry,
    SelectRoute,
    SubmitAuthorization,
    parse_action,
)

# Bump on ANY change to the prompt template, tool catalog, or parser (spec standard 1).
SCAFFOLD_VERSION = "1.0.0"

# The 9 tool models, keyed by their `tool` Literal — the real action space (contract.py).
_TOOL_MODELS = (
    SelectRoute, ComputeTax, SubmitAuthorization, HandleChallenge, Retry,
    ModifyCart, RequestUserConfirmation, Capture, Abort,
)


def _tool_name(tool_schema: dict[str, object]) -> str:
    """Extract the `tool` Literal value from its JSON-schema fragment.

    A single-value `Literal[...]` may serialize as `{"const": "..."}` or `{"enum": ["..."]}`
    depending on the pydantic version; accept either so the catalog never breaks on an upgrade.
    """
    if "const" in tool_schema:
        return str(tool_schema["const"])
    enum = tool_schema["enum"]
    assert isinstance(enum, list)
    return str(enum[0])


def tool_catalog() -> str:
    """A stable JSON string mapping each tool name -> its arg JSON-schema.

    Derived from the contract pydantic models via `model_json_schema()`, so the catalog can never
    drift from the real action space. The `tool` discriminator field is stripped from `properties`
    and `required` (the agent supplies it as the top-level "tool" key, not inside args).
    """
    catalog: dict[str, dict[str, object]] = {}
    for model in _TOOL_MODELS:
        schema = model.model_json_schema()
        props = {k: v for k, v in schema.get("properties", {}).items() if k != "tool"}
        # The Literal["..."] on `tool` is the dict key.
        name = _tool_name(schema["properties"]["tool"])
        required = [r for r in schema.get("required", []) if r != "tool"]
        catalog[name] = {"properties": props, "required": required}
    return json.dumps(catalog, sort_keys=True)


def render_observation(obs: Observation) -> str:
    """Render the redacted observation as a stable JSON string for the transcript.

    Uses pydantic json mode so Decimal money fields serialize as strings (never floats) and the
    output is deterministic for a given observation.
    """
    return obs.model_dump_json()


class CallUsage(BaseModel):
    """Per-response token + cost record, read by EM2's cost meter. Cost is float USD."""

    prompt_tokens: int
    completion_tokens: int
    response_cost: float | None


_OUTPUT_CONTRACT = (
    'Reply with EXACTLY ONE JSON object of the form {"tool": "<tool-name>", "args": {...}} and '
    "NOTHING else — no prose, no markdown fences. The available tools and their args schemas are:"
)

_REFORMAT_NUDGE = (
    "Your previous reply was not a single valid tool action. Reply with EXACTLY ONE JSON object "
    '{"tool": "<tool-name>", "args": {...}} from the catalog and nothing else.'
)


def _build_system_prompt(policy_text: str, mandate_text: str) -> str:
    return (
        f"{policy_text}\n\n"
        f"## Episode mandate\n{mandate_text}\n\n"
        f"## Output contract\n{_OUTPUT_CONTRACT}\n{tool_catalog()}"
    )


def _to_action(content: str) -> Action:
    """Parse a model reply ({"tool","args"}) into an Action. Raises on malformed/unknown/bad-args."""
    raw = json.loads(content)
    data = {"tool": raw["tool"], **raw.get("args", {})}
    return parse_action(data)


class ModelAgent:
    """Agent-under-test (spec §5.1). Satisfies `Agent.act`. One fresh instance per pass^k trial."""

    def __init__(
        self,
        *,
        route: str,
        policy_text: str,
        sampling: StageSampling,
        supports_response_format: bool,
        completion_fn: CompletionFn,
        mandate_text: str,
    ) -> None:
        self.route = route
        self.sampling = sampling
        self.supports_response_format = supports_response_format
        self._completion_fn = completion_fn
        self.usage: list[CallUsage] = []
        self.transcript: list[dict[str, str]] = [
            {"role": "system", "content": _build_system_prompt(policy_text, mandate_text)}
        ]

    def _sampling_kwargs(self) -> dict[str, object]:
        kw: dict[str, object] = {
            "temperature": self.sampling.temperature,
            "top_p": self.sampling.top_p,
            "max_tokens": self.sampling.max_tokens,
        }
        if self.sampling.seed is not None:
            kw["seed"] = self.sampling.seed
        if self.supports_response_format:
            kw["response_format"] = {"type": "json_object"}
        return kw

    def _call(self, messages: list[dict[str, str]]) -> str:
        # Pass a snapshot so a later transcript append cannot retroactively mutate the sent payload.
        resp = self._completion_fn(model=self.route, messages=list(messages), **self._sampling_kwargs())
        hidden = getattr(resp, "_hidden_params", {}) or {}
        cost = hidden.get("response_cost")
        usage = resp.usage
        self.usage.append(
            CallUsage(
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                response_cost=float(cost) if cost is not None else None,
            )
        )
        return str(resp.choices[0].message.content)

    def act(self, observation: Observation) -> Action:
        self.transcript.append({"role": "user", "content": render_observation(observation)})
        content = self._call(self.transcript)
        try:
            action = _to_action(content)
        except Exception:
            # ONE bounded "return only the JSON action" reformat retry (spec §5.1).
            retry_messages = [
                *self.transcript,
                {"role": "assistant", "content": content},
                {"role": "user", "content": _REFORMAT_NUDGE},
            ]
            retry_content = self._call(retry_messages)
            try:
                action = _to_action(retry_content)
                content = retry_content
            except Exception:
                # A model that cannot follow the contract scores honestly (malformed-rate published).
                action = Abort(tool="abort", reason="malformed_action")
                content = action.model_dump_json()
        self.transcript.append({"role": "assistant", "content": content})
        return action


def agent_factory(
    model: ModelConfig,
    *,
    policy_text: str,
    sampling: StageSampling,
    completion_fn: CompletionFn,
    mandate_text: str,
) -> AgentFactory:
    """Build a zero-arg factory of FRESH `ModelAgent`s — one fresh instance per pass^k trial."""

    def _make() -> ModelAgent:
        return ModelAgent(
            route=model.route,
            policy_text=policy_text,
            sampling=sampling,
            supports_response_format=model.supports_response_format,
            completion_fn=completion_fn,
            mandate_text=mandate_text,
        )

    return _make
