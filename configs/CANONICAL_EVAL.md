# Canonical Spar Eval Config (frontier runs MUST use this)

Dataset: build version `canonical-r1`, seed 1 (cooperative responder on non-traps; underspecified
escalate-then-complete; easy fraud_reactivity challenge-enabled). Canary recorded in the build manifest.

Invocation:
    PYTHONPATH=. .venv/bin/spar eval \
      --models configs/models.toml --profile configs/profile.toml \
      --dataset-dir build/ds/private \
      --responder-model openrouter/google/gemini-2.0-flash-001@2026-05 \
      --grader-model openrouter/google/gemini-2.0-flash-001@2026-05 \
      --out-dir runs_canonical --budget-usd <cap> --concurrency <n>

- responder-model: a cheap, capable, NEUTRAL model not under evaluation (gemini-2.0-flash-001,
  version-pinned @2026-05). It roleplays the principal; it is injection-defended (C4).
- grader-model: a pinned judge for the Tier-C semantic gray zone (model-graded surface <=10%,
  enforced by MODEL_GRADED_CAP). Omit only for offline smoke (stub grader is crude).
- Both model ids + the dataset canary are stamped into runs_canonical/<model>/run_manifest.json.
