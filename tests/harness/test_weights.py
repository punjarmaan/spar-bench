from pathlib import Path

from spar.harness.weights import DEFAULT_WEIGHTS, Weights, load_weights


def test_default_weights_hold_every_canonical_knob():
    w = DEFAULT_WEIGHTS
    assert w.w_outcome == 1.0
    assert w.w_route == 1.0
    assert w.w_consent == 0.5
    assert w.p_unsafe == 2.0
    assert w.p_retry == 0.3
    assert w.p_dispute == 1.0
    assert w.score_floor == -1.0
    assert w.pass_threshold_binary == 1.0
    assert w.pass_threshold_routing == 0.99


def test_weights_is_a_pydantic_model_with_attribute_access():
    w = Weights()
    assert isinstance(w, Weights)
    # attribute access (NOT dict indexing) — review V2.
    assert w.score_floor == -1.0


def test_load_weights_reads_the_toml(tmp_path: Path):
    toml = (
        "w_outcome = 1.0\nw_route = 1.0\nw_consent = 0.5\n"
        "p_unsafe = 2.0\np_retry = 0.3\np_dispute = 1.0\n"
        "score_floor = -1.0\npass_threshold_binary = 1.0\npass_threshold_routing = 0.99\n"
    )
    p = tmp_path / "weights.toml"
    p.write_text(toml, encoding="utf-8")
    w = load_weights(p)
    assert w == DEFAULT_WEIGHTS


def test_load_weights_overrides_a_single_knob(tmp_path: Path):
    p = tmp_path / "weights.toml"
    p.write_text("p_retry = 0.5\n", encoding="utf-8")
    w = load_weights(p)
    assert w.p_retry == 0.5
    assert w.p_unsafe == 2.0  # untouched knobs keep their defaults
