# Reproducibility anchor (module 50 §3). Frozen, no resolver drift.
#
# RELEASE NOTE: pin BOTH the base image and the uv installer BY DIGEST (sha256) on a
# deliberate spar_version bump and record the digests in the HF card. Tags are used here so
# the image is buildable in the dev/CI environment; a floating tag can be re-pushed, so the
# release build must replace them with immutable digests:
#   docker inspect --format='{{index .RepoDigests 0}}' python:3.11-slim
FROM python:3.11-slim

# uv: copy the static binary from the official image (pin BY DIGEST at release).
COPY --from=ghcr.io/astral-sh/uv:0.4.18 /uv /usr/local/bin/uv

WORKDIR /spar

# Copy the lockfile + project metadata first for layer caching; install FROZEN (no drift).
COPY pyproject.toml uv.lock README.md ./
COPY spar ./spar
COPY tests ./tests
RUN uv sync --frozen --extra dev

# Default reproducibility check: the deterministic validation gate (no model calls). A green
# gate in a clean frozen environment IS the "baselines reproduce" guarantee (F1/F10). The
# published per-split baselines are reproduced server-side from the frozen splits (the local
# CLI never serves the private split — C5).
ENTRYPOINT ["uv", "run", "--frozen"]
CMD ["pytest", "tests/test_baselines.py", "-q"]
