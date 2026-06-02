"""Tests for robust responder reply parsing in LiteLLMUserSim.

Covers:
- Happy path: clean numeric bound parses correctly (unchanged behavior).
- Tolerant bound coercion: "1,000.00", "$50", "50 USD" → correct Decimal.
- Unparseable bounds: "N/A", "" → None, no crash.
- Non-JSON first reply, valid JSON on re-prompt → re-prompted answer returned (called twice).
- Both replies unparseable → decision="deny" fallback, called twice, no crash.
- Re-prompt cost is recorded.
"""
from __future__ import annotations

from decimal import Decimal

import pytest

from spar.harness.user_sim import LiteLLMUserSim, UserResponse, UserSimRequest


# ---------------------------------------------------------------------------
# Helper: build a fake completion_fn that sequences through a list of reply
# content strings.  After the list is exhausted it raises to catch mistakes.
# ---------------------------------------------------------------------------

def _make_fake(*contents: str, cost: float = 0.001):
    """Return (fake_fn, call_count_dict).  Each call pops the next content."""
    seq = list(contents)
    counter = {"n": 0}

    def _fake(*, model, messages, temperature, response_format=None):
        counter["n"] += 1
        content = seq.pop(0)

        class _Msg:
            pass

        msg = _Msg()
        msg.content = content  # type: ignore[attr-defined]

        class _Choice:
            message = msg  # type: ignore[assignment]

        class _Hidden:
            response_cost = cost

        class _Resp:
            choices = [_Choice()]
            _hidden_params = {"response_cost": cost}

        return _Resp()

    return _fake, counter


# ---------------------------------------------------------------------------
# 1. Happy path: clean numeric bound (behavior must be identical to before)
# ---------------------------------------------------------------------------

def test_happy_path_clean_numeric_bound():
    fake, calls = _make_fake('{"decision": "approve_bound", "bound": "50.00", "message": "ok"}')
    sim = LiteLLMUserSim(model="m@1", completion_fn=fake)
    resp = sim.respond(UserSimRequest(goal="buy a lamp", reason="within budget?"))
    assert isinstance(resp, UserResponse)
    assert resp.decision == "approve_bound"
    assert resp.bound == Decimal("50.00")
    assert resp.message == "ok"
    assert calls["n"] == 1


def test_happy_path_approve_no_bound():
    fake, calls = _make_fake('{"decision": "approve", "message": "go ahead"}')
    sim = LiteLLMUserSim(model="m@1", completion_fn=fake)
    resp = sim.respond(UserSimRequest(goal="buy coffee", reason="under $20"))
    assert resp.decision == "approve"
    assert resp.bound is None
    assert calls["n"] == 1


def test_happy_path_deny():
    fake, calls = _make_fake('{"decision": "deny", "message": "no"}')
    sim = LiteLLMUserSim(model="m@1", completion_fn=fake)
    resp = sim.respond(UserSimRequest(goal="buy coffee", reason="expensive"))
    assert resp.decision == "deny"
    assert resp.bound is None
    assert calls["n"] == 1


# ---------------------------------------------------------------------------
# 2. Tolerant bound coercion
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw_bound,expected", [
    ("1,000.00", Decimal("1000.00")),
    ("$50", Decimal("50")),
    ("50 USD", Decimal("50")),
    ("$ 1,234.56", Decimal("1234.56")),
])
def test_tolerant_bound_coercion(raw_bound, expected):
    content = f'{{"decision": "approve_bound", "bound": "{raw_bound}", "message": "ok"}}'
    fake, calls = _make_fake(content)
    sim = LiteLLMUserSim(model="m@1", completion_fn=fake)
    resp = sim.respond(UserSimRequest(goal="buy a blender", reason="ok?"))
    assert resp.decision == "approve_bound"
    assert resp.bound == expected, f"For raw_bound={raw_bound!r}: expected {expected}, got {resp.bound}"
    assert calls["n"] == 1  # no re-prompt needed


# ---------------------------------------------------------------------------
# 3. Unparseable bounds → None, decision preserved, no crash
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw_bound", ["N/A", "", "unknown"])
def test_unparseable_bound_becomes_none(raw_bound):
    content = f'{{"decision": "approve_bound", "bound": "{raw_bound}", "message": "ok"}}'
    fake, calls = _make_fake(content)
    sim = LiteLLMUserSim(model="m@1", completion_fn=fake)
    resp = sim.respond(UserSimRequest(goal="buy a blender", reason="ok?"))
    assert resp.bound is None
    assert resp.decision == "approve_bound"
    assert calls["n"] == 1  # no re-prompt needed


def test_null_bound_becomes_none():
    """Explicit JSON null bound stays None."""
    fake, calls = _make_fake('{"decision": "approve", "bound": null, "message": "sure"}')
    sim = LiteLLMUserSim(model="m@1", completion_fn=fake)
    resp = sim.respond(UserSimRequest(goal="buy a blender", reason="ok?"))
    assert resp.bound is None
    assert resp.decision == "approve"


# ---------------------------------------------------------------------------
# 4. Non-JSON first reply, valid JSON on re-prompt → re-prompted answer returned
# ---------------------------------------------------------------------------

def test_non_json_first_reply_reprompted():
    first = "Sorry, I can't answer that right now."
    second = '{"decision": "approve_bound", "bound": "75", "message": "ok after reprompt"}'
    fake, calls = _make_fake(first, second)
    sim = LiteLLMUserSim(model="m@1", completion_fn=fake)
    resp = sim.respond(UserSimRequest(goal="buy a TV", reason="expensive?"))
    assert calls["n"] == 2, "Should have called completion_fn twice (initial + re-prompt)"
    assert resp.decision == "approve_bound"
    assert resp.bound == Decimal("75")
    assert resp.message == "ok after reprompt"


def test_non_json_first_reply_reprompt_is_cached():
    """After re-prompt succeeds, result is cached (third call for same request = 0 new calls)."""
    first = "oops not json"
    second = '{"decision": "deny", "message": "no"}'
    fake, calls = _make_fake(first, second)
    sim = LiteLLMUserSim(model="m@1", completion_fn=fake)
    req = UserSimRequest(goal="buy a TV", reason="expensive?")
    sim.respond(req)
    # Second call for same request should hit cache
    sim.respond(req)
    assert calls["n"] == 2, "Cache should prevent a third model call"


# ---------------------------------------------------------------------------
# 5. Both replies unparseable → deny fallback, called twice, no crash
# ---------------------------------------------------------------------------

def test_both_replies_unparseable_returns_deny():
    fake, calls = _make_fake("not json at all", "also not json")
    sim = LiteLLMUserSim(model="m@1", completion_fn=fake)
    resp = sim.respond(UserSimRequest(goal="buy earphones", reason="confirm?"))
    assert calls["n"] == 2
    assert resp.decision == "deny"
    assert resp.bound is None
    assert resp.message is not None  # some diagnostic message


def test_first_reply_bad_json_second_reply_bad_json_no_crash():
    """No exception must escape even when both completions return garbage."""
    fake, calls = _make_fake("{broken json", '{"decision": "INVALID_VALUE"}')
    sim = LiteLLMUserSim(model="m@1", completion_fn=fake)
    resp = sim.respond(UserSimRequest(goal="buy headphones", reason="ok?"))
    assert calls["n"] == 2
    assert resp.decision == "deny"


# ---------------------------------------------------------------------------
# 6. Re-prompt cost is recorded
# ---------------------------------------------------------------------------

def test_reprompt_cost_is_recorded():
    first = "not json"
    second = '{"decision": "deny", "message": "no"}'
    fake, calls = _make_fake(first, second, cost=0.005)
    sim = LiteLLMUserSim(model="m@1", completion_fn=fake)
    sim.respond(UserSimRequest(goal="buy shoes", reason="ok?"))
    # Two calls each at $0.005 = $0.010
    assert calls["n"] == 2
    assert abs(sim.cost_usd - 0.010) < 1e-9, f"Expected 0.010, got {sim.cost_usd}"


def test_reprompt_cost_on_deny_fallback():
    """Even when deny fallback fires (both bad), costs from both calls are recorded."""
    fake, calls = _make_fake("garbage", "more garbage", cost=0.003)
    sim = LiteLLMUserSim(model="m@1", completion_fn=fake)
    sim.respond(UserSimRequest(goal="buy shoes", reason="ok?"))
    assert abs(sim.cost_usd - 0.006) < 1e-9, f"Expected 0.006, got {sim.cost_usd}"


# ---------------------------------------------------------------------------
# 7. First reply is valid JSON but missing required 'decision' key → re-prompt
# ---------------------------------------------------------------------------

def test_missing_decision_key_triggers_reprompt():
    first = '{"bound": "50", "message": "oops no decision"}'
    second = '{"decision": "approve_bound", "bound": "50", "message": "fixed"}'
    fake, calls = _make_fake(first, second)
    sim = LiteLLMUserSim(model="m@1", completion_fn=fake)
    resp = sim.respond(UserSimRequest(goal="buy a lamp", reason="ok?"))
    assert calls["n"] == 2
    assert resp.decision == "approve_bound"
    assert resp.bound == Decimal("50")
