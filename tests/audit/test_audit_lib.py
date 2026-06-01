from pathlib import Path
from scripts.audit_lib import load_all_samples, cell_key, ADMITTED_RUNS, load_clean_outcomes

DATASET = Path("build/ds/private")

def test_load_all_samples_counts():
    by_split = {}
    for s in load_all_samples(DATASET):
        by_split[s.split] = by_split.get(s.split, 0) + 1
    assert by_split == {"main": 602, "lite": 63, "diamond": 4}

def test_cell_key_is_stable_tuple():
    s = load_all_samples(DATASET)[0]
    assert cell_key(s) == (s.axis.value, s.difficulty.value, s.is_trap, s.intent_spec.value)

def test_admitted_runs_excludes_contaminated():
    paths = {r["dir"] for r in ADMITTED_RUNS}
    assert "runs_main/deepseek-v3.2" not in paths
    assert "runs_main/gpt-oss-120b-free" in paths

def test_load_clean_outcomes_main_has_two_models():
    out = load_clean_outcomes()
    sid = [s.sample_id for s in load_all_samples(DATASET) if s.split == "main"][0]
    rows = out.get(sid, [])
    assert len(rows) >= 2
    assert {"gemini-2.0-flash", "gpt-oss-120b-free"}.issubset({o["model"] for o in rows})
    assert set(rows[0]) >= {"model", "status", "final_state", "score", "is_trap",
                            "catastrophic_class", "unsafe_completion", "split"}
