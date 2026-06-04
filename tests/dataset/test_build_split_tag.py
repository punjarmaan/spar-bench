"""Build stamps Sample.split onto the private full-gold build only; public rows are unaffected."""

from __future__ import annotations

import json
from pathlib import Path

from spar.dataset.build import build


def _rows(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def test_build_stamps_split_on_private_not_public(tmp_path: Path) -> None:
    public_dir = tmp_path / "public"
    private_dir = tmp_path / "private"
    build(public_dir=public_dir, private_dir=private_dir, build_seed=7, spar_version="0.1.0")

    private_rows = _rows(private_dir / "private.jsonl")
    assert private_rows, "private build is non-empty"
    # every private sample carries a valid split tag
    assert all(r["split"] in {"lite", "main", "redline"} for r in private_rows)
    # each procedural + backbone split is actually present (a dropped split would slip past `all`)
    assert any(r["split"] == "lite" for r in private_rows)
    assert any(r["split"] == "main" for r in private_rows)
    assert any(r["split"] == "redline" for r in private_rows)

    # public rows NEVER carry the tag (projection allowlist + private-only stamping)
    public_main = _rows(public_dir / "main.jsonl")
    assert public_main, "public main split is non-empty"
    assert all("split" not in r for r in public_main)
