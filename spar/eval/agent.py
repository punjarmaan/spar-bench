"""The agent-under-test, driven by a uniform JSON-action protocol.

One code path for every model. Native tool-calling is not used: it isn't uniform across the open
models, which would confound the comparison. The system prompt is policy + mandate + a tool catalog
derived from the `contract` pydantic models (so it never drifts) + the JSON output contract. Each
`act` appends the rendered observation to an instance transcript, calls the injected `completion_fn`,
json.loads -> parse_action -> Action; one bounded reformat retry; else Abort(malformed_action).

litellm is not imported here — the completion_fn is injected. Cost is float USD; Decimal is reserved
for simulator money.
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING

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
    Refund,
    RequestUserConfirmation,
    Retry,
    SelectRoute,
    SubmitAuthorization,
    Void,
    parse_action,
)

if TYPE_CHECKING:
    from spar.eval.trajectory import AgentTurn  # runtime import stays lazy in act() (circular)

# Bump on ANY change to the prompt template, tool catalog, or parser.
SCAFFOLD_VERSION = "2.2.0"

# Native reasoning effort level passed to the provider for reasoning-capable models.
# litellm-standard level ("low"/"medium"/"high"); pinned to "high". Changing this value bumps the
# frozen scaffold — increment SCAFFOLD_VERSION.
REASONING_EFFORT = "high"

# Output-token cap for reasoning calls. Reasoning tokens count toward max_tokens, and effort=high
# overruns the profile's 2048 (truncates to empty content -> malformed) on complex turns. Generous
# cap; only consumes tokens if the model actually reasons that far.
REASONING_MAX_TOKENS = 16384

# The 11 tool models, keyed by their `tool` Literal — the real action space.
_TOOL_MODELS = (
    SelectRoute, ComputeTax, SubmitAuthorization, HandleChallenge, Retry,
    ModifyCart, RequestUserConfirmation, Capture, Void, Refund, Abort,
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


# Valid tool names derived from _TOOL_MODELS — used by the lenient fallback parser so it never
# drifts from the real action space.
_VALID_TOOLS: frozenset[str] = frozenset(
    _tool_name(model.model_json_schema()["properties"]["tool"])
    for model in _TOOL_MODELS
)


def render_observation(obs: Observation) -> str:
    """Render the redacted observation as a stable JSON string for the transcript.

    Uses pydantic json mode so Decimal money fields serialize as strings (never floats) and the
    output is deterministic for a given observation.
    """
    return obs.model_dump_json()


class CallUsage(BaseModel):
    """Per-response token + cost record, read by the cost meter. Cost is float USD.

    `billed` is False for a cache hit (the completion was already paid for on the run that
    populated the cache): it counts toward the gross uncached cost but not the cost actually spent
    this run, and never trips the budget cap. A live cache-miss call is billed.
    """

    prompt_tokens: int
    completion_tokens: int
    response_cost: float | None
    billed: bool = True


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
    """Parse a model reply into an Action. Raises on malformed/unknown/bad-args.

    Primary path: {"tool": "<name>", "args": {...}}.
    Lenient fallbacks (tried in order when primary fails):
      a. Strip markdown fences (```[json]...```) and retry primary path.
      b. Alternate envelope: a dict with exactly ONE key that is a valid tool name,
         e.g. {"compute_tax": {}} or {"request_user_confirmation": {"reason": "x"}}.
    Genuinely garbage input (prose, empty, unknown single key) still raises.
    """
    # --- Primary path ---
    def _parse_standard(raw_obj: dict) -> Action:  # type: ignore[return]
        data = {"tool": raw_obj["tool"], **raw_obj.get("args", {})}
        return parse_action(data)

    try:
        raw = json.loads(content)
        return _parse_standard(raw)
    except Exception as exc:
        primary_exc = exc

    # --- Fallback a: strip markdown fences ---
    stripped = content.strip()
    if stripped.startswith("```"):
        first_newline = stripped.find("\n")
        if first_newline != -1:
            inner = stripped[first_newline + 1:]
            if inner.rstrip().endswith("```"):
                inner = inner.rstrip()[:-3].rstrip()
            try:
                raw = json.loads(inner)
                return _parse_standard(raw)
            except Exception:
                pass

    # --- Fallback b: alternate envelope {tool_name: args_dict} ---
    try:
        raw = json.loads(content)
        if isinstance(raw, dict) and len(raw) == 1:
            (key, value) = next(iter(raw.items()))
            if key in _VALID_TOOLS:
                args = value if isinstance(value, dict) else {}
                data = {"tool": key, **args}
                return parse_action(data)
    except Exception:
        pass

    # All fallbacks exhausted — re-raise the original failure.
    raise primary_exc


def _debug_log_malformed(
    *, route: str, step: int, initial_content: str, retry_content: str,
    initial_reasoning: str | None, retry_reasoning: str | None,
) -> None:
    """Diagnostic only: append the raw replies that failed to parse to the path in the
    SPAR_DEBUG_MALFORMED env var. No-op when the var is unset — never affects a run's behavior or
    scoring; exists to inspect why reasoning models malform on the prompt-format path. Best-effort
    append; swallows IO errors so it can never break an episode."""
    path = os.environ.get("SPAR_DEBUG_MALFORMED")
    if not path:
        return
    rec = {
        "route": route, "step": step,
        "initial_content": initial_content[:4000], "retry_content": retry_content[:4000],
        "initial_reasoning": (initial_reasoning or "")[:4000],
        "retry_reasoning": (retry_reasoning or "")[:4000],
    }
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
    except OSError:
        pass


class ModelAgent:
    """Agent-under-test. Satisfies `Agent.act`. One fresh instance per pass^k trial."""

    def __init__(
        self,
        *,
        route: str,
        policy_text: str,
        sampling: StageSampling,
        supports_response_format: bool,
        completion_fn: CompletionFn,
        mandate_text: str,
        reasoning: bool = False,
    ) -> None:
        self.route = route
        self.sampling = sampling
        self.supports_response_format = supports_response_format
        self.reasoning = reasoning
        self._completion_fn = completion_fn
        self.usage: list[CallUsage] = []
        self.turns: list["AgentTurn"] = []  # enriched per-turn records
        self._last_reasoning: str | None = None   # reasoning trace from the most recent _call
        self.transcript: list[dict[str, str]] = [
            {"role": "system", "content": _build_system_prompt(policy_text, mandate_text)}
        ]

    def _sampling_kwargs(self) -> dict[str, object]:
        kw: dict[str, object] = {"max_tokens": self.sampling.max_tokens}
        # Native Anthropic/OpenAI thinking rejects fixed temperature/top_p/seed (drop_params won't
        # strip them) — omit there; every other route keeps the profile sampling.
        _native_reasoning = self.reasoning and (
            self.route.startswith("anthropic/") or self.route.startswith("openai/")
        )
        if not _native_reasoning:
            kw["temperature"] = self.sampling.temperature
            kw["top_p"] = self.sampling.top_p
            if self.sampling.seed is not None:
                kw["seed"] = self.sampling.seed
        if self.supports_response_format:
            kw["response_format"] = {"type": "json_object"}
        if self.reasoning:
            # reasoning_effort: pinned level, maps to provider-native thinking. include_reasoning:
            # OpenRouter-only (without it some OR providers return content=None; native APIs reject it
            # and drop_params won't strip it). drop_params: lets litellm drop sampling params a
            # reasoning model rejects.
            kw["reasoning_effort"] = REASONING_EFFORT
            if self.route.startswith("openrouter/"):
                kw["include_reasoning"] = True
            kw["drop_params"] = True
            # Reasoning tokens count toward max_tokens; effort=high overruns the profile's 2048
            # (truncates to empty content). Raise the cap — only consumed if the model reasons that far.
            kw["max_tokens"] = REASONING_MAX_TOKENS
        return kw

    def _call(self, messages: list[dict[str, str]]) -> str:
        # Pass a snapshot so a later transcript append cannot retroactively mutate the sent payload.
        resp = self._completion_fn(model=self.route, messages=list(messages), **self._sampling_kwargs())
        hidden = getattr(resp, "_hidden_params", {}) or {}
        cost = hidden.get("response_cost")
        usage = resp.usage
        # A cache replay was already paid for on a prior run: count it as gross-only, not billed.
        billed = not getattr(resp, "cache_hit", False)
        self.usage.append(
            CallUsage(
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                response_cost=float(cost) if cost is not None else None,
                billed=billed,
            )
        )
        msg = resp.choices[0].message
        self._last_reasoning = getattr(msg, "reasoning_content", None)
        raw_content = msg.content
        return "" if raw_content is None else str(raw_content)

    def act(self, observation: Observation) -> Action:
        from spar.eval.trajectory import AgentTurn  # lazy: trajectory.py imports CallUsage from here

        turn_index = len(self.turns)
        usage_start = len(self.usage)
        obs_json = render_observation(observation)
        self.transcript.append({"role": "user", "content": obs_json})
        content = self._call(self.transcript)
        initial_reasoning = self._last_reasoning
        initial_content = content
        retried = False
        retry_content: str | None = None
        retry_reasoning: str | None = None
        try:
            action = _to_action(content)
        except Exception:
            # One bounded "return only the JSON action" reformat retry.
            retried = True
            retry_messages = [
                *self.transcript,
                {"role": "assistant", "content": content},
                {"role": "user", "content": _REFORMAT_NUDGE},
            ]
            retry_content = self._call(retry_messages)
            retry_reasoning = self._last_reasoning
            try:
                action = _to_action(retry_content)
                content = retry_content
            except Exception:
                # A model that cannot follow the contract scores honestly (malformed-rate published).
                _debug_log_malformed(
                    route=self.route, step=len(self.transcript),
                    initial_content=initial_content, retry_content=retry_content,
                    initial_reasoning=initial_reasoning, retry_reasoning=retry_reasoning,
                )
                action = Abort(tool="abort", reason="malformed_action")
                content = action.model_dump_json()
        self.transcript.append({"role": "assistant", "content": content})
        self.turns.append(AgentTurn(
            index=turn_index,
            observation=json.loads(obs_json),
            reasoning=initial_reasoning,
            raw_output=initial_content,
            retried=retried,
            retry_raw_output=retry_content,
            retry_reasoning=retry_reasoning,
            # Normalized parsed action {tool, args}, consistent across lenient envelopes and
            # malformed-abort. raw_output keeps the verbatim model reply.
            action={"tool": action.tool, "args": action.model_dump(mode="json", exclude={"tool"})},
            usage=list(self.usage[usage_start:]),
        ))
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
            reasoning=model.reasoning,
        )

    return _make
