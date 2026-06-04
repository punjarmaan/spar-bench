# Freeze contracts

One file per published dataset version (e.g. `canonical-r1.md`). Each pins the exact, immutable
surfaces a published Spar leaderboard depends on, so a campaign can't be silently invalidated.

**Why.** A leaderboard compares models on an *identical* test. From the first publication run
through the last, five surfaces must stay constant:

1. **Samples** — version + canary + build_seed + sample-id hashes
2. **Grader code** — `graders.py` / `report.py`
3. **Profile** — `profile.toml` (k, temperatures, weights, split set)
4. **Model pins** — the responder + grader models (they move scores)
5. **Published scope** — which splits go public / on the leaderboard

Changing any one of them forces a **new version** + a **full re-run of every model**.

**Lifecycle of a version's contract:**

- `CANDIDATE` — built, undergoing the pre-freeze validation gate (cheap models on main+redline).
- `FROZEN` — validation passed; publication runs in progress / done. Do not touch frozen surfaces.
- `SUPERSEDED` — a defect forced a re-cut; a later version replaces it. Keep the file for provenance.

See [`../CANONICAL_EVAL.md`](../CANONICAL_EVAL.md) for the invocation.
