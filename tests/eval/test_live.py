"""EM4 — offline tests for the live agent completion_fn wiring (no real model calls)."""

from __future__ import annotations


def test_default_completion_fn_returns_callable() -> None:
    # Importing spar.eval.live must NOT require litellm; only calling
    # default_completion_fn() resolves it. If litellm is unavailable the call
    # raises ImportError — but in the dev/llm environment it returns a callable.
    from spar.eval.live import default_completion_fn

    fn = default_completion_fn()
    assert callable(fn)


def test_live_module_imports_without_calling_litellm() -> None:
    # The module-level import is lazy: importing it must never touch litellm.
    import importlib

    mod = importlib.import_module("spar.eval.live")
    assert hasattr(mod, "default_completion_fn")
    # Nothing at module scope may have bound litellm.completion eagerly.
    assert "litellm" not in dir(mod)
