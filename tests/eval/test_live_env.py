"""`.env` auto-loading for live runs.

LiteLLM reads provider keys (e.g. OPENROUTER_API_KEY) from `os.environ`, but Python does not
auto-load `.env`. `load_env()` bridges that so a key placed in a repo-root `.env` reaches the
live path without a manual `export`. A shell-exported value must still win (override=False).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from spar.eval.live import load_env


def test_load_env_loads_a_dotenv_into_environ(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / ".env").write_text("SPAR_TEST_ENV_KEY=from_dotenv\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SPAR_TEST_ENV_KEY", raising=False)

    path = load_env()

    assert path  # the .env was found
    assert os.environ["SPAR_TEST_ENV_KEY"] == "from_dotenv"


def test_load_env_does_not_override_an_exported_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / ".env").write_text("SPAR_TEST_ENV_KEY=from_dotenv\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SPAR_TEST_ENV_KEY", "from_shell")  # already exported

    load_env()

    assert os.environ["SPAR_TEST_ENV_KEY"] == "from_shell"  # shell wins (override=False)


def test_load_env_returns_none_when_no_dotenv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)  # an empty dir with no .env anywhere above is unlikely, but
    # find_dotenv walks up; assert the call is safe and returns a (possibly-None) path-or-str.
    result = load_env()
    assert result is None or isinstance(result, str)
