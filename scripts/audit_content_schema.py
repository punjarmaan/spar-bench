"""Schema gate for behavior-blind L3 content verdicts emitted by Claude judge subagents."""
from __future__ import annotations

CONTENT_VERDICTS = {"meaningful", "trivial", "ambiguous", "misspecified", "meaningless"}


def validate_content_verdict(row: dict) -> dict:
    for field in ("sample_id", "content_verdict", "rationale"):
        if field not in row:
            raise ValueError(f"content verdict missing field: {field!r}")
    if row["content_verdict"] not in CONTENT_VERDICTS:
        raise ValueError(f"unknown content_verdict: {row['content_verdict']!r}")
    return row
