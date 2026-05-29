"""M7 Task 12: load frozen split artifacts (F13/F17) — public projected, private full graded."""

from __future__ import annotations

import pytest

from spar.dataset.build import build
from spar.dataset.loader import load_public_split, load_split
from spar.simulator.schemas import Sample


def _build(tmp_path):
    pub = tmp_path / "public"
    priv = tmp_path / "private"
    build(public_dir=pub, private_dir=priv, build_seed=12345, spar_version="1.0.0")
    return pub, priv


def test_load_public_split_reads_projected_frozen_artifact(tmp_path):
    pub, _ = _build(tmp_path)
    rows = load_public_split("main", base_dir=pub)
    assert len(rows) >= 1
    assert all("gold" not in r for r in rows)
    assert all(r["canary"].startswith("spar:") for r in rows)
    assert all(r["sample_id"].startswith("spar_") for r in rows)


def test_load_private_split_reads_full_graded_samples(tmp_path):
    _, priv = _build(tmp_path)
    samples = load_split("private", base_dir=priv)
    assert samples and all(isinstance(s, Sample) for s in samples)
    assert all(s.gold is not None for s in samples)


def test_load_split_rejects_public_split_with_base_dir(tmp_path):
    pub, _ = _build(tmp_path)
    with pytest.raises(ValueError):
        load_split("main", base_dir=pub)  # public splits are projected; use load_public_split


def test_load_split_rejects_unknown_split(tmp_path):
    with pytest.raises(ValueError):
        load_split("bogus", base_dir=tmp_path)
