"""Model registry (spec §5.2): model IDENTITY only — never sampling knobs (those live in the profile).

A frozen pydantic `ModelConfig` plus `load_models` reading `[[model]]` tables from a TOML file.
`class` is a python reserved word, so the field is `cls` with serialization alias "class".
Cost prices are float USD (reporting metadata; Decimal is reserved for simulator money).
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ModelConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    id: str                       # display + filename-safe key
    route: str                    # LiteLLM model string, e.g. "openrouter/anthropic/claude-opus-4"
    cls: Literal["frontier", "open"] = Field(alias="class")   # serialized as "class"
    supports_response_format: bool = True
    price_in_per_mtok: float | None = None
    price_out_per_mtok: float | None = None
    version_pin: str | None = None


def load_models(path: str | Path) -> list[ModelConfig]:
    """Read `[[model]]` tables from a TOML file into validated `ModelConfig`s (spec §5.2)."""
    data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    return [ModelConfig.model_validate(row) for row in data.get("model", [])]
