"""L1: deterministic, responder-independent per-sample structural flags (no live model calls)."""
from __future__ import annotations

import json
from pathlib import Path

from spar.agents.naive_complete import NaiveCompleteAgent
from spar.agents.reference_agents import AlwaysCompleteAgent, LateRefundAgent
from spar.harness.graders import score
from spar.harness.runner import run_episode
from spar.simulator.enums import ViolationClass
from spar.simulator.schemas import Sample

from scripts.audit_lib import DATASET_DIR, cell_key, load_all_samples

OUT = Path("build/audit/structural.jsonl")


def naive_passes(s: Sample) -> bool:
    """True iff NaiveCompleteAgent completes this non-trap correctly — a dataset-invariant breach
    (F1 says naive must FAIL every non-trap). Restricted to EXPLICIT non-traps: semantic/
    underspecified grading needs a model grader, which an offline L1 check does not have."""
    if s.is_trap or s.intent_spec.value != "explicit":
        return False
    return bool(score(s, run_episode(s, NaiveCompleteAgent(), trial_index=0)).outcome_correct)


def trap_trippable(s: Sample) -> bool:
    """A trap is trippable iff a maximally-violating reference agent produces an unsafe grade.
    Covers BOTH catastrophic-class traps and must_not (empty expected_violations) traps."""
    if not s.is_trap:
        return True
    agents = [AlwaysCompleteAgent()]
    if ViolationClass.MISSED_REVERSAL_WINDOW in s.gold.expected_violations:
        agents.append(LateRefundAgent())
    for agent in agents:
        sc = score(s, run_episode(s, agent, trial_index=0))
        if sc.unsafe_completion or sc.catastrophic_class is not None:
            return True
    return False


def structural_flags(s: Sample) -> list[str]:
    flags: list[str] = []
    if set(s.gold.must) & set(s.gold.must_not):
        flags.append("must_contradiction")
    if s.is_trap and not trap_trippable(s):
        # AlwaysCompleteAgent/LateRefundAgent are build-proven adversaries ONLY for catastrophic
        # classes. For must_not traps (empty expected_violations) a non-trip is INCONCLUSIVE (the
        # generic adversary may not perform the specific forbidden action), so flag it weaker and
        # route it to content review rather than asserting the trap is broken.
        if s.gold.expected_violations:
            flags.append("untrippable_trap")
        else:
            flags.append("trippability_unverified")
    if (not s.is_trap and s.intent_spec.value == "explicit"
            and getattr(s.mandate, "amount_limit", None) is None):
        flags.append("missing_scenario_field")
    if s.is_trap and bool(s.gold.expected_violations) and not s.world_config.issuer_behavior:
        flags.append("missing_scenario_field")
    if naive_passes(s):
        flags.append("naive_passes")
    return flags


def run(dataset_dir: Path = DATASET_DIR) -> dict[str, list[str]]:
    samples = load_all_samples(dataset_dir)
    flags = {s.sample_id: structural_flags(s) for s in samples}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as fh:
        for s in samples:
            fh.write(json.dumps({"sample_id": s.sample_id, "split": s.split,
                                 "cell": list(cell_key(s)), "l1_flags": flags[s.sample_id]}) + "\n")
    return flags


if __name__ == "__main__":
    f = run()
    print(f"L1 wrote {OUT} | {sum(1 for v in f.values() if v)}/{len(f)} samples carry >=1 flag")
