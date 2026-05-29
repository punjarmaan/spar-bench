from spar.dataset.loader import load_split


def test_load_toy_lite_split_returns_validated_samples():
    samples = load_split("lite")
    assert len(samples) >= 1
    assert samples[0].sample_id.startswith("spar_")
    assert samples[0].canary.startswith("spar:")
