# Canonical Spar Eval Config (frontier runs MUST use this)

Dataset: build version `canonical-r1`, seed 1 (cooperative responder on non-traps; underspecified
escalate-then-complete; easy fraud_reactivity challenge-enabled). Canary recorded in the build manifest.

Invocation:
    PYTHONPATH=. .venv/bin/spar eval \
      --models configs/models.toml --profile configs/profile.toml \
      --dataset-dir build/ds/private \
      --responder-model openrouter/google/gemini-2.5-flash \
      --grader-model openrouter/google/gemini-2.5-flash \
      --out-dir runs_canonical --cache-dir .eval_cache_canon --budget-usd <cap> --concurrency <n>

- responder-model: a cheap, capable, NEUTRAL model not under evaluation (`google/gemini-2.5-flash`).
  It roleplays the principal; it is injection-defended (C4).
- grader-model: a pinned judge for the Tier-C semantic gray zone (model-graded surface <=10%,
  enforced by MODEL_GRADED_CAP). Omit only for offline smoke (stub grader is crude).
- Both model ids + the dataset canary are stamped into runs_canonical/<model>/run_manifest.json.

NOTE — model ids (verified against the live OpenRouter catalog, 2026-06):
- OpenRouter has NO `@date` pin syntax. Pass BARE ids (e.g. `openrouter/google/gemini-2.5-flash`).
  A `...@2026-05` suffix returns HTTP 400 "not a valid model ID".
- `google/gemini-2.0-flash-001` was DELISTED (404). The prior pin to it was the root cause of a
  validation-run failure where every escalation 404'd and the harness mislabeled it `context_overflow`.
- Re-verify every roster route resolves on OpenRouter before a paid run (a 404 is silently scored as
  a capability abort). As of 2026-06 the five frontier routes (claude-opus-4, gpt-5, claude-sonnet-4,
  deepseek-v3.2, qwen3-235b-a22b-2507) all resolve.
- Always pass a fresh `--cache-dir` for a clean campaign so a prior failed run can't replay stale state.
