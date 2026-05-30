"""Local graded-split resolution: the Sample.split tag + load_graded_split resolver."""

from __future__ import annotations

from spar.dataset.loader import load_split


def test_sample_split_defaults_none_and_round_trips() -> None:
    sample = load_split("lite")[0]
    assert sample.split is None                       # additive field defaults to None
    tagged = sample.model_copy(update={"split": "main"})
    assert tagged.split == "main"
    # survives a JSON round-trip (this is how it rides into private.jsonl)
    again = type(sample).model_validate_json(tagged.model_dump_json())
    assert again.split == "main"
