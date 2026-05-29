import json

from spar.eval.agent import SCAFFOLD_VERSION, tool_catalog


def test_scaffold_version_is_pinned():
    assert SCAFFOLD_VERSION == "1.0.0"


def test_tool_catalog_is_json_covering_all_nine_tools():
    cat = tool_catalog()
    assert isinstance(cat, str)
    parsed = json.loads(cat)  # must be valid JSON so it never drifts from the contract
    # Catalog maps each tool name -> its JSON arg schema, derived from contract models.
    names = set(parsed)
    assert names == {
        "select_route", "compute_tax", "submit_authorization", "handle_challenge",
        "retry", "modify_cart", "request_user_confirmation", "capture", "abort",
    }
    # Args schema is derived from the pydantic models (so it cannot drift).
    assert "acquirer_id" in parsed["select_route"]["properties"]
    assert "method" in parsed["select_route"]["properties"]
    assert "reason" in parsed["abort"]["properties"]
    assert "strategy" in parsed["retry"]["properties"]
    # compute_tax / capture take no args beyond the discriminator.
    assert set(parsed["compute_tax"]["properties"]) == set()


def test_tool_catalog_is_deterministic():
    assert tool_catalog() == tool_catalog()
