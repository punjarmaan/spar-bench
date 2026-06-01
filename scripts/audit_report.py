"""Render the read-only audit report from sample_verdicts.jsonl (robust verdicts only)."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

SRC = Path("build/audit/sample_verdicts.jsonl")
OUT = Path("docs/superpowers/reports/2026-06-01-spar-sample-validity-audit.md")


def go_no_go(rows: list[dict]) -> dict:
    main = [r for r in rows if r["split"] == "main"]
    n = len(main) or 1
    keep = sum(1 for r in main if r["verdict"] in ("KEEP", "WEAK"))
    cut = sum(1 for r in main if r["verdict"] == "CUT")
    keep_frac = keep / n
    recommend = "RUN" if (keep_frac >= 0.9 and cut <= 0.02 * n) else "FIX_FIRST"
    return {"recommend": recommend, "keep_frac": keep_frac, "n_main": len(main),
            "cut": cut, "fix": sum(1 for r in main if r["verdict"] == "FIX")}


def render_report(rows: list[dict]) -> str:
    by_split: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        by_split[r["split"]][r["verdict"]] += 1
    nt_refuse = sum(1 for r in rows if r["split"] == "main"
                    and r.get("l2_context", {}).get("nontrap_all_refuse"))
    L = ["# Spar Sample Validity Audit", "", "## Per-split verdicts", "",
         "| split | KEEP | WEAK | FIX | CUT |", "|---|---|---|---|---|"]
    for sp in ("main", "lite", "diamond"):
        c = by_split[sp]
        L.append(f"| {sp} | {c['KEEP']} | {c['WEAK']} | {c['FIX']} | {c['CUT']} |")
    g = go_no_go(rows)
    L += ["", "## Go / No-Go", "",
          f"- main KEEP+WEAK: **{g['keep_frac']:.1%}** (FIX={g['fix']}, CUT={g['cut']}, n={g['n_main']})",
          f"- **Recommendation: {g['recommend']}**",
          f"- Context: {nt_refuse} main non-traps are 'all-refuse' — this is the **always-deny** "
          "responder signature (a config artifact, NOT a defect rate); it does not drive any verdict.",
          ""]
    # FIX/CUT candidates (may be empty)
    L += ["## FIX / CUT candidates", "",
          "| sample_id | split | verdict | concerns | content |", "|---|---|---|---|---|"]
    fixcut = [r for r in rows if r["verdict"] in ("FIX", "CUT")]
    if not fixcut:
        L.append("| _(none)_ | | | | |")
    for r in sorted(fixcut, key=lambda x: (x["verdict"] != "CUT", x["split"])):
        cc = "; ".join(f"{c[0]}:{c[2]}" for c in r["concerns"])
        L.append(f"| {r['sample_id']} | {r['split']} | {r['verdict']} | {cc} | {r.get('content_verdict')} |")
    # Review queue: WEAK items (single-layer flags, not condemned)
    weak = [r for r in rows if r["verdict"] == "WEAK"]
    by_concern: dict[str, list[dict]] = defaultdict(list)
    for r in weak:
        for c in r["concerns"]:
            by_concern[c[2]].append(r)
    L += ["", "## Review queue (WEAK — single-layer flags, surfaced not condemned)", "",
          f"{len(weak)} samples carry a single-layer flag (capped at WEAK by the corroboration rule).", ""]
    for code in sorted(by_concern):
        items = by_concern[code]
        L.append(f"### {code} ({len(items)})")
        # show content-flagged ones explicitly with sample_ids; for bulk structural flags, summarize
        if code in ("misspecified", "meaningless", "ambiguous"):
            for r in items:
                L.append(f"- `{r['sample_id']}` ({r['split']}) — content judge: {r.get('content_verdict')}")
        else:
            ids = ", ".join(f"`{r['sample_id']}`" for r in items[:40])
            L.append(f"- {ids}{' …' if len(items) > 40 else ''}")
        L.append("")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    rows = [json.loads(l) for l in SRC.read_text().splitlines() if l.strip()]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(render_report(rows), encoding="utf-8")
    print(f"wrote {OUT} | go/no-go: {go_no_go(rows)['recommend']}")
