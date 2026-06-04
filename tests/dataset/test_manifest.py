import hashlib

from spar.simulator.enums import Axis, Difficulty, IntentSpec
from spar.dataset.generator import GenSpec, generate
from spar.dataset.manifest import build_manifest


def _samples():
    specs = [
        GenSpec(axis=Axis.ROUTING, seed=1, difficulty=Difficulty.EASY, is_trap=False,
                intent_spec=IntentSpec.EXPLICIT),
        GenSpec(axis=Axis.ROUTING, seed=2, difficulty=Difficulty.HARD, is_trap=True,
                intent_spec=IntentSpec.EXPLICIT),
    ]
    return [generate(s) for s in specs]


def test_manifest_has_all_required_fields():
    m = build_manifest(_samples(), split="main", build_seed=12345,
                       canary="spar:t", spar_version="1.0.0")
    for key in ("split", "spar_version", "schema_version", "build_seed", "canary",
                "n_samples", "counts", "n_redline", "n_model_graded",
                "trap_fraction", "sample_ids_sha256"):
        assert key in m
    assert m["n_samples"] == 2


def test_counts_breakdown_is_axis_difficulty_trap():
    m = build_manifest(_samples(), split="main", build_seed=1, canary="spar:t",
                       spar_version="1.0.0")
    routing = m["counts"]["routing"]
    assert routing["easy"]["non_trap"] == 1
    assert routing["hard"]["trap"] == 1


def test_sample_ids_sha256_is_digest_of_sorted_ids():
    samples = _samples()
    m = build_manifest(samples, split="main", build_seed=1, canary="spar:t",
                       spar_version="1.0.0")
    expected = hashlib.sha256(
        "\n".join(sorted(s.sample_id for s in samples)).encode("utf-8")
    ).hexdigest()
    assert m["sample_ids_sha256"] == expected


def test_trap_fraction_is_realized():
    m = build_manifest(_samples(), split="main", build_seed=1, canary="spar:t",
                       spar_version="1.0.0")
    assert m["trap_fraction"] == 0.5  # 1 of 2
