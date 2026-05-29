from spar.simulator.rng import stable_hash, derive_seed, SubStream, substream


def test_stable_hash_is_process_independent():
    # Not Python's salted hash(): same input → same output every run.
    assert stable_hash("spar_x_0001") == stable_hash("spar_x_0001")
    assert isinstance(stable_hash("a"), int)


def test_derive_seed_is_pure_function_of_inputs():
    assert derive_seed(123, 0) == derive_seed(123, 0)
    assert derive_seed(123, 0) != derive_seed(123, 1)


def test_named_substreams_are_independent_and_reproducible():
    g1 = substream("spar_x_0001", seed=42, trial_index=0, stream=SubStream.DECLINE, step=0)
    g2 = substream("spar_x_0001", seed=42, trial_index=0, stream=SubStream.DECLINE, step=0)
    assert g1.random() == g2.random()  # reproducible
    g3 = substream("spar_x_0001", seed=42, trial_index=0, stream=SubStream.CAPTURE, step=0)
    # different stream → (almost surely) different draw
    assert substream("spar_x_0001", seed=42, trial_index=0, stream=SubStream.DECLINE, step=0).random() != g3.random()
