"""Substream-layer determinism proof (foundation-revision Task 5.2, correction C10).

C10 supersedes the plan's original `noop_prefix` approach (REJECTED: there is no
universally-legal no-op). Instead we prove the two REAL determinism properties of the
World as a deterministic function of (sample, trial_index, action sequence):

(a) PURE REPLAY — the SAME action sequence through two fresh `World`s yields byte-identical
    `hidden_final_state` and final FSM state.

(b) ORDINAL-KEYING — two worlds reach the SAME logical transition (the auth resolution) via
    DIFFERENT legal action counts: path B inserts extra legal `ComputeTax` actions before
    `submit_authorization`. `ComputeTax` bumps `elapsed_steps` but does NOT bump the auth
    ordinal (`auth_attempts[acquirer]` / `_auth_attempt`), so the pinned auth DRAW is keyed
    on the stable transition ordinal, NEVER on `elapsed_steps`. The extra ComputeTax shifts
    `elapsed_steps` but must NOT shift the draw.
"""

from __future__ import annotations

import pytest

from spar.simulator.contract import (
    Capture,
    ComputeTax,
    SelectRoute,
    SubmitAuthorization,
)
from spar.simulator.enums import ToolStatus
from spar.simulator.world import World
from tests.simulator._world_fixtures import build_world


# --------------------------------------------------------------------------- (a)


def _drive_settle(w: World) -> None:
    w.step(SelectRoute(tool="select_route", acquirer_id="a1", method="card_visa"))
    w.step(SubmitAuthorization(tool="submit_authorization", idempotency_key="k1"))
    w.step(Capture(tool="capture", idempotency_key="k2"))


@pytest.mark.parametrize("approval_prob", [1.0, 0.5, 0.02])
def test_pure_replay_identical_hidden_and_terminal(approval_prob: float) -> None:
    """Same sample + trial_index + action sequence -> identical hidden state and terminal."""
    w1 = build_world(approval_prob=approval_prob)
    _drive_settle(w1)
    w2 = build_world(approval_prob=approval_prob)
    _drive_settle(w2)

    assert w1.hidden_final_state == w2.hidden_final_state
    assert w1.state == w2.state
    # Draining the grade-terminal must also be deterministic.
    assert w1.drain_deferred() == w2.drain_deferred()
    assert w1.hidden_final_state == w2.hidden_final_state


# --------------------------------------------------------------------------- (b)


# approval_prob=0.02 -> the auth DRAW at ordinal 0 DECLINES (reason 05): a genuine,
# non-constant RNG outcome (not a trivial always-approve). 0.5 -> approves; both must be
# identical across the two action-count paths regardless of which outcome they land on.
@pytest.mark.parametrize("approval_prob", [0.02, 0.5])
@pytest.mark.parametrize("extra_compute_tax", [1, 2, 3])
def test_extra_legal_action_does_not_shift_auth_draw(
    approval_prob: float, extra_compute_tax: int
) -> None:
    """Path A: select -> submit. Path B: select -> N*compute_tax -> submit.

    `ComputeTax` is legal from ROUTE_SELECTED and does NOT bump the auth ordinal, so both
    submits resolve at the SAME stable auth ordinal (auth_attempts['a1'] == 0 when read) ->
    identical draw. The extra ComputeTax only shifts `elapsed_steps` (C10 ordinal-keying).
    """
    wa = build_world(approval_prob=approval_prob)
    wa.step(SelectRoute(tool="select_route", acquirer_id="a1", method="card_visa"))
    ra = wa.step(SubmitAuthorization(tool="submit_authorization", idempotency_key="k1"))

    wb = build_world(approval_prob=approval_prob)
    wb.step(SelectRoute(tool="select_route", acquirer_id="a1", method="card_visa"))
    for _ in range(extra_compute_tax):
        rt = wb.step(ComputeTax(tool="compute_tax"))
        assert rt.status is ToolStatus.OK  # ComputeTax is a legal observe from ROUTE_SELECTED
    rb = wb.step(SubmitAuthorization(tool="submit_authorization", idempotency_key="k1"))

    # The action counts genuinely differ (the extra ComputeTax shifted the clock)...
    assert wa.elapsed_steps != wb.elapsed_steps
    assert wb.elapsed_steps == wa.elapsed_steps + extra_compute_tax
    # ...but the auth ordinal is identical (keyed on the stable transition, not the clock)...
    assert wa.auth_attempts == wb.auth_attempts == {"a1": 1}
    # ...so the auth DRAW is byte-identical: same status AND same decline reason_code.
    assert ra.status == rb.status
    assert ra.reason_code == rb.reason_code
    # And the post-submit FSM state matches too.
    assert wa.state == wb.state


def test_auth_draw_is_observably_non_trivial() -> None:
    """Guard against a vacuous (b): at approval_prob=0.02 the auth draw must actually DECLINE.

    If this asserted approval the ordinal-keying proof would be a constant-outcome no-op;
    the decline confirms the RNG was genuinely exercised at the auth ordinal.
    """
    w = build_world(approval_prob=0.02)
    w.step(SelectRoute(tool="select_route", acquirer_id="a1", method="card_visa"))
    r = w.step(SubmitAuthorization(tool="submit_authorization", idempotency_key="k1"))
    assert r.status is ToolStatus.DECLINED
    assert r.reason_code == "05"
