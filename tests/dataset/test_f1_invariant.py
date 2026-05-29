from spar.simulator.enums import Axis, Difficulty, IntentSpec
from spar.dataset.generator import GenSpec, generate
from spar.agents.naive_complete import NAIVE_COMPLETE_TOOLS


def test_every_non_trap_must_token_is_outside_naive_toolset():
    # F1: a bare select->submit->capture must NOT satisfy any non-trap's must list.
    for axis in Axis:
        for diff in (Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD):
            s = generate(GenSpec(axis=axis, seed=1, difficulty=diff, is_trap=False,
                                 intent_spec=IntentSpec.EXPLICIT))
            assert s.gold.must, f"{axis}/{diff} non-trap has empty must (F1 violation)"
            # No must token is satisfiable by the naive toolset alone.
            assert not (set(s.gold.must) & NAIVE_COMPLETE_TOOLS)


def test_naive_toolset_is_exactly_the_three_trivial_tools():
    assert NAIVE_COMPLETE_TOOLS == frozenset(
        {"select_route", "submit_authorization", "capture"}
    )
