#!/usr/bin/env bash
# Docker reproducibility smoke-test (module 50 §3/§4). Build the frozen image and run the
# deterministic validation gate inside it. A green gate in a clean, frozen environment is the
# "baselines reproduce" guarantee (F1/F10) — no resolver drift, no model calls.
set -euo pipefail

IMAGE="spar:repro-smoke"
docker build -f docker/Dockerfile -t "$IMAGE" .
docker run --rm "$IMAGE" pytest tests/test_baselines.py -q
echo "REPRO OK"
