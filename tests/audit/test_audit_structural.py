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
