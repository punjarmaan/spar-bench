# Freeze Contract — `canonical-r2`

- **Status:** `CANDIDATE` — built 2026-06-03 with the full construct-validity remediation (Plans A+B).
  NOT frozen. The first-freeze decision is the user's; do NOT flip to FROZEN without explicit go.
- **Supersedes:** `canonical-r1` (canary `780f5805…`, retired — its procedural samples were not
  construct-valid; see `docs/superpowers/specs/2026-06-02-dataset-construct-validity-remediation.md`).

## Samples (dataset identity)
| field | value |
|---|---|
| `spar_version` | `canonical-r2` |
| `canary` | `spar:054b5bcc-b1ea-4180-8e61-aa1b04d60e96` |
| `build_seed` | `1` |
| private `n_samples` | `669` |
| private `sample_ids_sha256` | `368e2927f3513be1a68461f9170b5e94d9249f355fc4d0e78b12392b89d92d08` |

Build: `PYTHONPATH=. .venv/bin/spar build --seed 1 --public-out build/ds/public --private-out build/ds/private --version canonical-r2`
(sample_ids are stable vs r1; the WORLD/GOLD CONTENT changed — new canary is the identity field.)

## What changed vs r1 (construct validity)
- **Per-axis trap mechanisms (Issue 1):** routing (unsupported-route), decline_recovery (real hard decline),
  stale_state (price drift), fraud_reactivity (guaranteed soft block) — no longer the universal over-limit trap.
- **Diversity (Issue 4):** 597/602 distinct world-config signatures (was ~29); 4 currencies, 5 binding geos,
  3 instrument families, 8 MCCs, 8 categories; amounts straddle limits; ≥3 behavioral realizations per catastrophic class.
- **Honesty:** no inert `must_not` tokens; consent over-limit/scope-wall floor (11 traps).
- **Grader (Plan A):** catastrophic zeroing decoupled (full-set) from per-class attribution (`gold.intended_violation_class`);
  published headline = `trust_score_useful` (refusal-aware); always-escalate gaming-floor baseline.

## Enforced build gates (the construct-validity guarantee)
On `main`+`private`: `f1_spotcheck`, `safe_completion_spotcheck`, `trap_mechanism_spotcheck`,
`must_not_reachable_spotcheck`, `coverage_spotcheck` (intended-keyed, 7/7), consent over-limit floor (≥10).
`solvability_spotcheck` is LOG-ONLY (the reference DiligentAgent is a safe baseline, not a competent solver;
dynamic solvability is validated by live model runs).

## Reliability (Issue 6)
Headline reliability = AGGREGATE pass^4 + CI; per-axis pass^4 is a wide-CI diagnostic only.
NOTE: diamond all-7-axis expansion (Plan C / Task C1) is DEFERRED — current diamond covers 3/7 axes
(consent×2 + the procedural-tagged set if added). Expand before any publication run.

## Unchanged from r1 (machinery)
`graders.py` per-episode scoring shape, `report.py` formulas, `profile.toml`, responder/grader pins
(`openrouter/google/gemini-2.5-flash`), scaffold_version `2.2.0`.

## Decline-code taxonomy (review M1 — already correct)
`spar/simulator/reasons.py`: `43`/`41`/`62` hard (no-retry), soft codes retriable — documented as correct.
