"""The live agent call layer.

`default_completion_fn()` lazily imports `litellm` and returns `litellm.completion`. This is the
production `completion_fn` for the agent under test when `spar eval` injects no fake. Model routes
are plain LiteLLM model strings — any LiteLLM-supported provider works (e.g.
"openrouter/<provider>/<model>", "anthropic/<model>", "openai/<model>") — and LiteLLM resolves the
provider and reads its API key (OPENROUTER_API_KEY, ANTHROPIC_API_KEY, …) from the environment
automatically — no key plumbing here.

The import is lazy so `spar/eval/` stays importable without the optional `llm` extra; only an actual
live run resolves litellm.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

CompletionFn = Callable[..., Any]


def load_env() -> str | None:
    """Load a repo-root `.env` into `os.environ` so a key placed there (e.g. OPENROUTER_API_KEY,
    ANTHROPIC_API_KEY, or any other provider's key) is visible to LiteLLM, which reads provider
    keys from the environment.

    `python-dotenv` ships as a LiteLLM dependency, so the `llm` extra already provides it; if it
    is somehow absent this is a no-op (the user can still export the key manually). `override=False`
    means a variable already exported in the shell wins over `.env`. Returns the resolved `.env`
    path (for logging), or None if none was found / python-dotenv is unavailable.
    """
    try:
        from dotenv import find_dotenv, load_dotenv
    except ImportError:  # pragma: no cover - only without python-dotenv
        return None
    path = find_dotenv(usecwd=True)
    if path:
        load_dotenv(path, override=False)
        return path
    return None


def default_completion_fn() -> CompletionFn:
    """Return `litellm.completion`, importing litellm lazily on first call.

    Raises ImportError (with the install hint) if the optional `llm` extra is absent.
    """
    try:
        import litellm
    except ImportError as exc:  # pragma: no cover - exercised only without the llm extra
        raise ImportError(
            "litellm is required for live model calls. Install it with: "
            'uv pip install -e ".[llm]"  (and set the API key for your model routes\' '
            "provider, e.g. OPENROUTER_API_KEY)."
        ) from exc
    return litellm.completion  # type: ignore[no-any-return]
