import pytest
from scripts.audit_content_schema import validate_content_verdict, CONTENT_VERDICTS

def test_valid_passes():
    assert validate_content_verdict(
        {"sample_id":"x","content_verdict":"meaningful","rationale":"clear"})["content_verdict"] == "meaningful"

def test_unknown_rejected():
    with pytest.raises(ValueError):
        validate_content_verdict({"sample_id":"x","content_verdict":"great","rationale":"y"})

def test_missing_field_rejected():
    with pytest.raises(ValueError):
        validate_content_verdict({"sample_id":"x","content_verdict":"meaningful"})

def test_taxonomy():
    assert CONTENT_VERDICTS == {"meaningful","trivial","ambiguous","misspecified","meaningless"}
