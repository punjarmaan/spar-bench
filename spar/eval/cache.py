"""Content-addressed completion cache (design §5.5/§5.6): resume + reproducibility.

Keys hash (model, messages, sampling) via `stable_hash`; each entry is a JSON file under
`dir`. A cache hit lets a re-run skip the model call and reproduce identical grades. The
`digest` summarizes the populated cache for the run manifest. Stdlib only — no litellm.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from spar.simulator.rng import stable_hash


def cache_key(model: str, messages: list[dict[str, Any]], sampling: dict[str, Any]) -> str:
    """Stable hex key for one completion request. Same inputs -> same key (cross-process)."""
    payload = json.dumps(
        {"model": model, "messages": messages, "sampling": sampling},
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return f"{stable_hash(payload):016x}"


class CompletionCache:
    """One JSON file per cached completion, keyed by `cache_key`. Survives process restarts."""

    def __init__(self, dir: Path) -> None:
        self.dir = Path(dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.dir / f"{key}.json"

    def get(self, key: str) -> dict[str, Any] | None:
        path = self._path(key)
        if not path.is_file():
            return None
        result: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return result

    def put(self, key: str, response: dict[str, Any]) -> None:
        self._path(key).write_text(
            json.dumps(response, sort_keys=True, default=str), encoding="utf-8"
        )

    def digest(self) -> str:
        """A stable digest of the populated key set (for run_manifest reproducibility)."""
        keys = sorted(p.stem for p in self.dir.glob("*.json"))
        return f"{stable_hash(json.dumps(keys)):016x}"
