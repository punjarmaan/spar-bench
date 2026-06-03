#!/usr/bin/env python
"""Build a deterministic, stratified ~50% subset of `main` as its OWN split: "probe".

Operates on RAW LINES — parses only to stratify. The selected rows are emitted with
their `split` field re-tagged "main"->"probe" via a single exact-token string replace
(NOT a json.dumps round-trip, so Decimal serialization is preserved byte-for-byte).
Full diamond/lite are retained verbatim so the dataset-dir stays self-contained.

"probe" is a derived capability-probe split, distinct from canonical published main.

Run: PYTHONPATH=. .venv/bin/python scripts/curate_main_probe.py
"""

import hashlib
import json
from collections import defaultdict
from pathlib import Path

SRC = Path("build/ds/private/private.jsonl")
OUT = Path("build/ds/private_main_probe/private.jsonl")
FRACTION = 0.50
SEED = "main-probe-r1"  # change to reshuffle; deterministic given this string

lines = [line for line in SRC.read_text().splitlines() if line.strip()]
parsed = [(line, json.loads(line)) for line in lines]

main = [(line, r) for line, r in parsed if r.get("split") == "main"]
other = [line for line, r in parsed if r.get("split") in ("diamond", "lite")]  # keep full


def cat_applicable(r):
    return bool((r.get("gold") or {}).get("expected_violations"))


def stratum(r):
    return (r["axis"], bool(r["is_trap"]), cat_applicable(r), r["difficulty"])


buckets = defaultdict(list)
for line, r in main:
    buckets[stratum(r)].append((line, r))

SPLIT_TOKEN = '"split":"main"'
PROBE_TOKEN = '"split":"probe"'

selected = []
for key, items in sorted(buckets.items(), key=lambda kv: str(kv[0])):
    # deterministic seeded order, then take round(FRACTION*n) (>=1 if non-empty)
    items_sorted = sorted(
        items,
        key=lambda lr: hashlib.sha256(f"{SEED}:{lr[1]['sample_id']}".encode()).hexdigest(),
    )
    n_keep = max(1, round(FRACTION * len(items)))
    for line, r in items_sorted[:n_keep]:
        # re-tag main -> probe with a single exact-token replace (verbatim otherwise).
        assert line.count(SPLIT_TOKEN) == 1, f"expected exactly one {SPLIT_TOKEN} in {r['sample_id']}"
        selected.append(line.replace(SPLIT_TOKEN, PROBE_TOKEN, 1))

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text("\n".join(other + selected) + "\n")
print(f"probe (re-tagged from main) {len(selected)}/{len(main)} | + diamond/lite {len(other)} | total {len(other)+len(selected)}")
