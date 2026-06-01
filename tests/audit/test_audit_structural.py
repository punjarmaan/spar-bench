from pathlib import Path
from scripts.audit_lib import load_all_samples
from scripts.audit_structural import structural_flags, naive_passes, trap_trippable

DATASET = Path("build/ds/private")

def test_flags_over_all_samples_are_lists():
    flags = {s.sample_id: structural_flags(s) for s in load_all_samples(DATASET)}
    assert len(flags) == 669 and all(isinstance(v, list) for v in flags.values())

def test_no_must_contradiction_in_dataset():
    for s in load_all_samples(DATASET):
        assert "must_contradiction" not in structural_flags(s)

def test_naive_passes_flags_nontraps_only():
    flagged = [s.sample_id for s in load_all_samples(DATASET) if naive_passes(s)]
    assert all("_trap_" not in sid for sid in flagged)

def test_trap_trippable_returns_bool_and_nontraps_are_trippable():
    for s in load_all_samples(DATASET):
        assert isinstance(trap_trippable(s), bool)
        if not s.is_trap:
            assert trap_trippable(s) is True

def test_untrippable_only_for_catastrophic_else_unverified():
    for s in load_all_samples(DATASET):
        fl = structural_flags(s)
        if "untrippable_trap" in fl:
            assert s.gold.expected_violations, f"{s.sample_id}: untrippable_trap needs catastrophic EV"
        if "trippability_unverified" in fl:
            assert s.is_trap and not s.gold.expected_violations
