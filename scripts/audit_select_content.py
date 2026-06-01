"""Choose the tiered/cell-aware set for L3 + build BEHAVIOR-BLIND judge payloads."""
from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_behavioral import behavioral_signals
from scripts.audit_lib import DATASET_DIR, cell_key, load_all_samples, load_clean_outcomes
from scripts.audit_structural import structural_flags

OUT = Path("build/audit/content_todo.jsonl")


def _payload(s) -> dict:
    m = s.mandate
    return {
        "is_trap": s.is_trap, "intent_spec": s.intent_spec.value,
        "axis": s.axis.value, "difficulty": s.difficulty.value,
        "mandate": {"goal": getattr(m, "goal", None),
                    "amount_limit": str(getattr(m, "amount_limit", None)),
                    "currency": getattr(m, "currency", None)},
        "issuer_behavior": s.world_config.issuer_behavior,
        "gold": {"correct_outcome": s.gold.correct_outcome.value,
                 "must": list(s.gold.must), "must_not": list(s.gold.must_not),
                 "expected_violations": [v.value for v in s.gold.expected_violations]},
        # NOTE: deliberately NO clean_behavior — the corroborating verdict must be behavior-blind.
    }


def select_for_content(dataset_dir: Path = DATASET_DIR) -> list[dict]:
    samples = load_all_samples(dataset_dir)
    outcomes = load_clean_outcomes()
    chosen: dict[str, dict] = {}
    cells_done: set = set()
    for s in samples:                                   # pass 1: every flagged sample
        l1 = structural_flags(s)
        rows = outcomes.get(s.sample_id, [])
        l2 = behavioral_signals(rows) if rows else {}
        reasons = list(l1) + (["malformed_magnet"] if l2.get("malformed_magnet") else [])
        if reasons:
            chosen[s.sample_id] = {"sample_id": s.sample_id, "cell": list(cell_key(s)),
                                   "reason": reasons, "payload": _payload(s)}
            cells_done.add(cell_key(s))
    for s in sorted(samples, key=lambda x: x.sample_id):  # pass 2: one rep per remaining cell
        ck = cell_key(s)
        if ck not in cells_done:
            chosen[s.sample_id] = {"sample_id": s.sample_id, "cell": list(ck),
                                   "reason": ["cell_representative"], "payload": _payload(s)}
            cells_done.add(ck)
    return list(chosen.values())


if __name__ == "__main__":
    todo = select_for_content()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(json.dumps(r) for r in todo) + "\n", encoding="utf-8")
    print(f"wrote {OUT} | {len(todo)} samples queued for behavior-blind L3 critique")
