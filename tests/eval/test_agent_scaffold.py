import json
from spar.eval.agent import tool_catalog, SCAFFOLD_VERSION, _to_action
from spar.simulator.contract import Void, Refund, Capture


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
    assert SCAFFOLD_VERSION == "2.1.0"
