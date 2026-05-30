"""The live agent call layer (spec §5.1, §5.4, §10).

`default_completion_fn()` lazily imports `litellm` and returns `litellm.completion`,
mirroring `spar.harness.model_grader._default_completion_fn`. This is the production
`completion_fn` for the AGENT UNDER TEST when `spar eval` injects no fake. OpenRouter is
the default route: model strings look like "openrouter/<provider>/<model>", and LiteLLM
reads `OPENROUTER_API_KEY` from the environment automatically — no key plumbing here.

The import is lazy so `spar/eval/` stays importable without the optional `llm` extra; only
an ACTUAL live run resolves litellm (matching the H4 LiteLLMModelGrader pattern).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

CompletionFn = Callable[..., Any]


def default_completion_fn() -> CompletionFn:
    """Return `litellm.completion`, importing litellm lazily on first call.

    Raises ImportError (with the install hint) if the optional `llm` extra is absent.
    """
    try:
        import litellm
    except ImportError as exc:  # pragma: no cover - exercised only without the llm extra
        raise ImportError(
            "litellm is required for live model calls. Install it with: "
            'uv pip install -e ".[llm]"  (and set OPENROUTER_API_KEY).'
        ) from exc
    return litellm.completion  # type: ignore[no-any-return]
