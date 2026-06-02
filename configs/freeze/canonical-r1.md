# Freeze Contract — `canonical-r1`

- **Status:** `CANDIDATE` — pending the pre-freeze validation gate (a free model + a very cheap
  model on main+diamond). Flip to `FROZEN` once that passes with **zero** changes to the surfaces
  below. Do not start the frontier/publication runs until `FROZEN`.
- **Created:** 2026-06-01
- **Supersedes:** all pre-canonical builds (e.g. canary `spar:5422343d…`, `spar:27342c14…`) — never published.

A published Spar leaderboard compares models on an identical test. From the first publication run
through the last, the five surfaces below **must not change**. Any change to a frozen surface
invalidates cross-model comparability and forces a new version (`canonical-r2`) + a full re-run.

## 1. Samples (dataset identity)

| field | value |
|---|---|
| `spar_version` | `canonical-r1` |
| `canary` | `spar:780f5805-8e46-435c-abd5-977e0dfd4a66` |
| `build_seed` | `1` |
| `schema_version` | `1` |
| private `n_samples` | `669` |
| private `sample_ids_sha256` | `368e2927f3513be1a68461f9170b5e94d9249f355fc4d0e78b12392b89d92d08` |
| main (public) `n_samples` | `602` |
| main (public) `sample_ids_sha256` | `1a87fa2f15eed616221de8804a60f8960aac01f98a8ddac18fc7cc0e1139e97c` |

Build command (deterministic given the repo commit below + seed 1):

```
PYTHONPATH=. .venv/bin/spar build --seed 1 \
  --public-out build/ds/public --private-out build/ds/private --version canonical-r1
```

**Verify before any run:** the `canary` + `sample_ids_sha256` in `build/ds/private/private.manifest.json`
match this table. A re-cut regenerates the canary; a mismatch means the sample set changed.

## 2. Grader logic (scoring code)

Run all publication evals from this repo commit, or a descendant that does **not** touch the files below:

| ref | value |
|---|---|
| repo commit | `dfaa5cbef30382de12d0621470e5dd95e3b7f431` |
| `spar/harness/graders.py` blob | `f6217301dae45bf32ae05150934ddcef498e0291` |
| `spar/harness/report.py` blob | `e6c4b2bf49ffab819bbc5396d752849cc5dc0c74` |

## 3. Profile (what is measured)

| ref | value |
|---|---|
| `configs/profile.toml` blob | `340c06251b51be19a45a1b94ceffc4bbf66ee314` |

- **main** — k=1, competence stage, temp 0.0 — **published**
- **diamond** — k=4, reliability stage, temp 0.7 — **published**
- **lite** — k=1, competence — **not published** (dev smoke; runs but discarded). Drop from the run
  profile if you want to avoid paying for it on the frontier models.

## 4. Model pins (responder + grader — they move scores, so they're frozen)

- `--responder-model`: `openrouter/google/gemini-2.5-flash`
- `--grader-model`: `openrouter/google/gemini-2.5-flash`

> Updated 2026-06-02: the original pin `gemini-2.0-flash-001@2026-05` was invalid — the model was
> delisted on OpenRouter (404) and the `@date` suffix is rejected (400). A validation run mislabeled
> the resulting failures as `context_overflow`. Pass BARE ids; OpenRouter has no `@date` pinning.
> `gemini-2.5-flash` confirmed live and is neutral vs the models under test (Anthropic/OpenAI/DeepSeek/Qwen).

## 5. Published scope

- **Public artifact:** `build/ds/public/main.jsonl` + `build/ds/public/diamond.jsonl` (projected, answer-free).
- **Held out (never published):** `build/ds/private/` (gold + canary).
- **Leaderboard splits:** main (competence / pass^1 + safety metrics) + diamond (pass^4 reliability).

## Allowed without breaking the freeze

- **Add a new model** and run it against the surfaces above → joins the leaderboard.
- **Recover a failed run** (e.g. an OpenRouter weekly-limit 403 that cached malformed responses) by
  re-running that model with a **fresh `--cache-dir`** against the same surfaces.

## Forces a new version (`canonical-r2`) + full re-run of ALL models

- Any sample change (generator edit / re-cut) → new canary.
- Any change to `graders.py` / `report.py` per-episode scoring.
- Any change to `profile.toml` (k, temperature, weights, split set).
- Any change to the responder or grader model pin.

## Run-manifest cross-check (before publishing)

Every `runs_*/<model>/run_manifest.json` stamps `canary`, `grader_model`, `responder_model`,
`sampling`, `weights`. Confirm **all** published models carry:

```
canary        = spar:780f5805-8e46-435c-abd5-977e0dfd4a66
grader_model  = responder_model = openrouter/google/gemini-2.5-flash
```

The **canary is the authoritative dataset-identity field** (it round-trips from the build). Do NOT
cross-check on the run manifest's `build_seed`/`spar_version` — those stamp code defaults (observed
`build_seed=0`, `spar_version=0.1.0`), not the dataset's seed-1/`canonical-r1`. A model whose manifest
differs on canary or the model pins is not comparable — exclude it or re-run it.
