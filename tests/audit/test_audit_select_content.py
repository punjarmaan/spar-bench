from pathlib import Path
from scripts.audit_select_content import select_for_content
from scripts.audit_lib import load_all_samples, cell_key

def test_selection_covers_all_cells_and_payload_is_behavior_blind():
    todo = select_for_content(Path("build/ds/private"))
    assert {tuple(r["cell"]) for r in todo} == {cell_key(s) for s in load_all_samples()}
    r = todo[0]
    assert set(r) >= {"sample_id", "cell", "reason", "payload"}
    assert set(r["payload"]) >= {"is_trap", "intent_spec", "mandate", "gold"}
    assert "clean_behavior" not in r["payload"]   # behavior-BLIND for corroboration
