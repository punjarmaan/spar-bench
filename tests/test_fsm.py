from decimal import Decimal

from hypothesis import given, settings
from hypothesis import strategies as st

from spar.simulator.contract import (
    Abort,
    Capture,
    HandleChallenge,
    ModifyCart,
    Retry,
    RequestUserConfirmation,
    SelectRoute,
    SubmitAuthorization,
)
from spar.simulator.digest import trajectory_digest
from spar.simulator.enums import (
    Axis,
    Difficulty,
    FsmState,
    IntentSpec,
)
from spar.simulator.mandates import IntentMandate
from spar.simulator.schemas import Acquirer, Gold, Sample, WorldConfig
from spar.simulator.world import World

_DEFINED_STATES = set(FsmState)


def _sample(decline_plan: dict | None = None) -> Sample:
    acq = Acquirer(
        acquirer_id="acq_a", methods=["visa"], supported_geos=["US"],
        advertised_fee_bps=200, observed_approval_band="high",
        true_fee_bps=200, approval_prob=1.0, reliability=1.0,
    )
    return Sample(
        sample_id="spar_decline_recovery_prop", axis=Axis.DECLINE_RECOVERY,
        difficulty=Difficulty.EASY, is_trap=False, intent_spec=IntentSpec.EXPLICIT,
        diamond=False, model_graded=False, seed=99, canary="spar:t",
        world_config=WorldConfig(
            acquirers=[acq],
            decline_plan=decline_plan or {"mode": "scripted", "draws": []},
            settlement="sync", max_steps=50,
        ),
        mandate=IntentMandate(
            goal="buy", amount_limit=Decimal("100"), currency="USD",
            human_present=True, conditions={},
            per_txn_max=Decimal("200"), daily_remaining=Decimal("350"),
            merchant_constraint=["acme"], mcc_constraint=None,
            allowed_instruments=["visa"], session_ttl_steps=50,
            single_use_or_recurring="single_use", time_window=None,
        ),
        policy_id="default_v1", gold=Gold(correct_outcome=FsmState.CLOSED),
    )


def _action_from(tag: str):
    return {
        "select_route": SelectRoute(tool="select_route", acquirer_id="acq_a", method="visa"),
        "submit": SubmitAuthorization(tool="submit_authorization"),
        "challenge": HandleChallenge(tool="handle_challenge", challenge_token="chal_1"),
        "retry": Retry(tool="retry", strategy="same"),
        "modify": ModifyCart(tool="modify_cart", field="pan", value="x"),
        "confirm": RequestUserConfirmation(tool="request_user_confirmation", reason="x"),
        "capture": Capture(tool="capture"),
        "abort": Abort(tool="abort", reason="x"),
    }[tag]


_TAGS = st.sampled_from(
    ["select_route", "submit", "challenge", "retry", "modify", "confirm", "capture", "abort"]
)


@settings(max_examples=400, deadline=None)
@given(seq=st.lists(_TAGS, min_size=0, max_size=15))
def test_no_action_sequence_reaches_an_undefined_state(seq):
    world = World(_sample(), trial_index=0)
    world.reset()
    for tag in seq:
        if world.is_agent_terminal():
            break
        world.step(_action_from(tag))
        assert world.state in _DEFINED_STATES


@settings(max_examples=200, deadline=None)
@given(seq=st.lists(_TAGS, min_size=1, max_size=12))
def test_terminal_states_are_absorbing(seq):
    world = World(_sample(), trial_index=0)
    world.reset()
    reached_terminal = False
    for tag in seq:
        if world.is_agent_terminal():
            before = world.state
            world.step(_action_from(tag))
            assert world.state is before  # absorbing: no transition out
            reached_terminal = True
        else:
            world.step(_action_from(tag))
    # If a terminal was ever reached, the world must still be terminal at the end (absorbing).
    if reached_terminal:
        assert world.is_agent_terminal()


def test_determinism_snapshot_is_pinned():
    plan = {"mode": "scripted", "draws": [
        {"attempt": 1, "outcome": "decline", "reason": "05"},
        {"attempt": 2, "outcome": "approve"},
    ]}
    trajectory = [
        SelectRoute(tool="select_route", acquirer_id="acq_a", method="visa"),
        SubmitAuthorization(tool="submit_authorization"),
        Retry(tool="retry", strategy="same"),
        SubmitAuthorization(tool="submit_authorization"),
        Capture(tool="capture"),
    ]
    world = World(_sample(plan), trial_index=0)
    world.reset()
    digest_a = trajectory_digest(world, trajectory)
    world_b = World(_sample(plan), trial_index=0)
    world_b.reset()
    digest_b = trajectory_digest(world_b, trajectory)
    assert digest_a == digest_b  # reproducible
    # Pinned value: regenerate intentionally only on a deliberate FSM change (dataset version bump).
    assert digest_a == _PINNED_DIGEST


# Pin captured from the first green run of trajectory_digest (Step 4); paste the printed value here.
_PINNED_DIGEST = "786e09bab5ba3752216bcebc985c4cab"
