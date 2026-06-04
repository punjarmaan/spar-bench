"""Local graded-split resolution: the Sample.split tag + load_graded_split resolver."""

from __future__ import annotations

from pathlib import Path

import pytest

from spar.dataset.loader import load_graded_split, load_split


def _write_private(base_dir: Path, samples) -> None:
    base_dir.mkdir(parents=True, exist_ok=True)
    body = "\n".join(s.model_dump_json() for s in samples)
    (base_dir / "private.jsonl").write_text(body + "\n", encoding="utf-8")


def _tagged(split: str, sample_id: str):
    return load_split("lite")[0].model_copy(update={"split": split, "sample_id": sample_id})


def test_sample_split_defaults_none_and_round_trips() -> None:
    sample = load_split("lite")[0]
    assert sample.split is None                       # additive field defaults to None
    tagged = sample.model_copy(update={"split": "main"})
    assert tagged.split == "main"
    # survives a JSON round-trip (this is how it rides into private.jsonl)
    again = type(sample).model_validate_json(tagged.model_dump_json())
    assert again.split == "main"


def test_load_graded_split_filters_by_tag_with_gold_intact(tmp_path: Path) -> None:
    _write_private(tmp_path, [_tagged("main", "m1"), _tagged("main", "m2"), _tagged("redline", "d1")])
    main = load_graded_split("main", base_dir=tmp_path)
    assert sorted(s.sample_id for s in main) == ["m1", "m2"]
    assert main[0].gold is not None                       # full gold preserved (gradeable)
    redline = load_graded_split("redline", base_dir=tmp_path)
    assert [s.sample_id for s in redline] == ["d1"]


def test_load_graded_split_unknown_split_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        load_graded_split("bogus", base_dir=tmp_path)     # validated before touching disk


def test_load_graded_split_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_graded_split("main", base_dir=tmp_path)      # no private.jsonl


def test_load_graded_split_no_matching_tag_raises(tmp_path: Path) -> None:
    _write_private(tmp_path, [load_split("lite")[0]])       # split is None -> no "main" match
    with pytest.raises(ValueError):
        load_graded_split("main", base_dir=tmp_path)
