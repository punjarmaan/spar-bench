"""Merge L1+L2+L3 into a per-sample verdict. Only responder-INDEPENDENT signals corroborate;
deny-sensitive L2 keys are intentionally NOT read here (spec 2b / 6)."""
from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_lib import DATASET_DIR, cell_key, load_all_samples

OUT = Path("build/audit/sample_verdicts.jsonl")

_L1_DISQ = {"untrippable_trap", "must_contradiction"}
_L1_FIX = {"missing_scenario_field"}
_L1_LOW = {"naive_passes", "trippability_unverified"}
_L3_DISQ = {"meaningless"}
_L3_FIX = {"ambiguous", "misspecified"}
_L3_LOW = {"trivial"}


def synthesize_verdict(l1_flags: list[str], l2_signals: dict, content_verdict: str | None) -> dict:
    concerns: list[tuple[str, str, str]] = []
    for f in l1_flags:
        sev = "DISQ" if f in _L1_DISQ else "FIX" if f in _L1_FIX else "LOW" if f in _L1_LOW else None
        if sev:
            concerns.append(("L1", sev, f))
    if l2_signals.get("malformed_magnet"):              # ONLY robust L2 signal corroborates
        concerns.append(("L2", "FIX", "malformed_magnet"))
    if content_verdict in _L3_DISQ:
        concerns.append(("L3", "DISQ", content_verdict))
    elif content_verdict in _L3_FIX:
        concerns.append(("L3", "FIX", content_verdict))
    elif content_verdict in _L3_LOW:
        concerns.append(("L3", "LOW", content_verdict))

    layers = {c[0] for c in concerns}
    has_disq = any(c[1] == "DISQ" for c in concerns)
    fix_or_disq_layers = {c[0] for c in concerns if c[1] in ("FIX", "DISQ")}
    # CUT requires >=2 distinct layers, but LOW-severity concerns cannot act as a
    # corroborating layer — only FIX or DISQ concerns count toward corroboration.
    corroborating_layers = {c[0] for c in concerns if c[1] in ("FIX", "DISQ")}
    if has_disq and len(corroborating_layers) >= 2:
        verdict = "CUT"
    elif len(fix_or_disq_layers) >= 2:
        verdict = "FIX"
    elif concerns:
        verdict = "WEAK"
    else:
        verdict = "KEEP"
    return {"verdict": verdict, "concerns": [list(c) for c in concerns], "layers": sorted(layers)}


def run(dataset_dir: Path = DATASET_DIR) -> dict[str, dict]:
    def _load(p):
        return {json.loads(l)["sample_id"]: json.loads(l)
                for l in Path(p).read_text().splitlines() if l.strip()}
    l1 = _load("build/audit/structural.jsonl")
    l2 = _load("build/audit/behavioral.jsonl")
    l3p = Path("build/audit/content_verdicts.jsonl")
    l3 = ({json.loads(l)["sample_id"]: json.loads(l)["content_verdict"]
           for l in l3p.read_text().splitlines() if l.strip()} if l3p.exists() else {})
    out = {}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as fh:
        for s in load_all_samples(dataset_dir):
            sid = s.sample_id
            v = synthesize_verdict(l1.get(sid, {}).get("l1_flags", []),
                                   l2.get(sid, {}).get("l2", {}), l3.get(sid))
            v.update({"sample_id": sid, "split": s.split, "cell": list(cell_key(s)),
                      "content_verdict": l3.get(sid),
                      "l2_context": {k: l2.get(sid, {}).get("l2", {}).get(k)
                                     for k in ("nontrap_all_refuse", "never_triggered_trap",
                                               "unanimous_pass", "discriminates", "n_clean_models")}})
            out[sid] = v
            fh.write(json.dumps(v) + "\n")
    return out


if __name__ == "__main__":
    from collections import Counter
    print("verdict tally:", dict(Counter(v["verdict"] for v in run().values())))
