from spar.eval.models import ModelConfig


def test_model_config_defaults_and_class_alias_round_trip():
    cfg = ModelConfig(
        id="claude-opus-4",
        route="openrouter/anthropic/claude-opus-4",
        cls="frontier",
    )
    assert cfg.id == "claude-opus-4"
    assert cfg.route == "openrouter/anthropic/claude-opus-4"
    assert cfg.cls == "frontier"
    # Defaults per the frozen contract.
    assert cfg.supports_response_format is True
    assert cfg.price_in_per_mtok is None
    assert cfg.price_out_per_mtok is None
    assert cfg.version_pin is None
    # `class` is the serialized/external key; `cls` is the python field.
    dumped = cfg.model_dump(by_alias=True)
    assert dumped["class"] == "frontier"
    assert "cls" not in dumped
    # And it accepts the external "class" key on construction.
    cfg2 = ModelConfig.model_validate(
        {"id": "x", "route": "r", "class": "open"}
    )
    assert cfg2.cls == "open"


def test_model_config_is_frozen():
    import pytest

    cfg = ModelConfig(id="x", route="r", cls="open")
    with pytest.raises(Exception):
        cfg.id = "y"  # type: ignore[misc]
