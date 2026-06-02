import json
import pytest
from spar.eval.agent import tool_catalog, SCAFFOLD_VERSION, _to_action
from spar.simulator.contract import Void, Refund, Capture, ComputeTax, RequestUserConfirmation


def test_catalog_has_eleven_tools_including_void_and_refund():
    cat = json.loads(tool_catalog())
    assert set(cat) == {
        "select_route", "compute_tax", "submit_authorization", "handle_challenge",
        "retry", "modify_cart", "request_user_confirmation", "capture",
        "void", "refund", "abort",
    }


def test_catalog_surfaces_idempotency_key_on_mutating_tools():
    cat = json.loads(tool_catalog())
    for tool in ("submit_authorization", "capture", "retry", "void", "refund"):
        assert "idempotency_key" in cat[tool]["properties"], tool


def test_parser_accepts_void_refund_and_keyed_capture():
    assert isinstance(_to_action('{"tool":"void","args":{"idempotency_key":"k"}}'), Void)
    assert isinstance(_to_action('{"tool":"refund","args":{"idempotency_key":"k"}}'), Refund)
    cap = _to_action('{"tool":"capture","args":{"idempotency_key":"k3"}}')
    assert isinstance(cap, Capture) and cap.idempotency_key == "k3"


def test_scaffold_version_bumped():
    assert SCAFFOLD_VERSION == "2.2.0"


# ---------------------------------------------------------------------------
# Lenient fallback parser tests (TDD — these must PASS after implementation)
# ---------------------------------------------------------------------------

def test_standard_envelope_unchanged():
    """Standard {"tool":"compute_tax","args":{}} still parses identically."""
    action = _to_action('{"tool":"compute_tax","args":{}}')
    assert isinstance(action, ComputeTax)


def test_alternate_envelope_no_args():
    """{"compute_tax": {}} — tool-name-as-key with empty dict value."""
    action = _to_action('{"compute_tax": {}}')
    assert isinstance(action, ComputeTax)


def test_alternate_envelope_with_args():
    """{"request_user_confirmation": {"reason": "why"}} — tool-name-as-key with arg dict."""
    action = _to_action('{"request_user_confirmation": {"reason": "why"}}')
    assert isinstance(action, RequestUserConfirmation)
    assert action.reason == "why"


def test_fenced_json_standard_envelope():
    """```json {"tool":"capture","args":{"idempotency_key":"k"}} ``` — markdown-fenced."""
    content = '```json\n{"tool":"capture","args":{"idempotency_key":"k"}}\n```'
    action = _to_action(content)
    assert isinstance(action, Capture)
    assert action.idempotency_key == "k"


def test_fenced_json_no_lang_tag():
    """``` {"tool":"compute_tax","args":{}} ``` — fenced without language tag."""
    content = '```\n{"tool":"compute_tax","args":{}}\n```'
    action = _to_action(content)
    assert isinstance(action, ComputeTax)


def test_garbage_still_raises():
    """Prose / empty still raises (caller converts to malformed Abort)."""
    with pytest.raises(Exception):
        _to_action("I will compute the tax for you now.")
    with pytest.raises(Exception):
        _to_action("")


def test_unknown_single_key_still_raises():
    """{"reason": "x"} — single key but NOT a valid tool name → still raises."""
    with pytest.raises(Exception):
        _to_action('{"reason": "x"}')
