import uuid

from spar.dataset.splits import make_canary, apply_canary
from spar.simulator.enums import Axis, Difficulty, IntentSpec
from spar.dataset.generator import GenSpec, generate


def test_make_canary_is_fresh_random_and_prefixed():
    # PLANS-REVIEW M7: the canary is a FRESH random UUID per build, NOT a function
    # of build_seed — otherwise a leaked seed lets a trainer pre-compute + scrub it.
    c1 = make_canary()
    c2 = make_canary()
    assert c1 != c2, "canary must be fresh per call, not deterministic"
    assert c1.startswith("spar:")
    # the body parses as a uuid4 (random)
    parsed = uuid.UUID(c1.removeprefix("spar:"))
    assert parsed.version == 4


def test_apply_canary_stamps_every_sample():
    sample = generate(GenSpec(axis=Axis.ROUTING, seed=1, difficulty=Difficulty.EASY,
                              is_trap=False, intent_spec=IntentSpec.EXPLICIT))
    stamped = apply_canary(sample, "spar:abc-123")
    assert stamped.canary == "spar:abc-123"
    assert sample.canary == "spar:UNSET"  # original unchanged (returns a copy)


def test_no_assign_split_export():
    # The old layered second-hash mechanism is removed (PLANS-REVIEW M7).
    import spar.dataset.splits as splits_mod
    assert not hasattr(splits_mod, "assign_split")
