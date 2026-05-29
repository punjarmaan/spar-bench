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


def test_load_models_reads_model_tables(tmp_path):
    from spar.eval.models import load_models

    toml = (
        '[[model]]\n'
        'id = "claude-opus-4"\n'
        'route = "openrouter/anthropic/claude-opus-4"\n'
        'class = "frontier"\n'
        'supports_response_format = true\n'
        'price_in_per_mtok = 15.0\n'
        'price_out_per_mtok = 75.0\n'
        'version_pin = "anthropic/claude-opus-4@2026-xx"\n'
        '\n'
        '[[model]]\n'
        'id = "qwen3-coder"\n'
        'route = "openrouter/qwen/qwen3-coder"\n'
        'class = "open"\n'
        'supports_response_format = false\n'
    )
    path = tmp_path / "models.toml"
    path.write_text(toml, encoding="utf-8")

    models = load_models(path)
    assert [m.id for m in models] == ["claude-opus-4", "qwen3-coder"]
    assert models[0].cls == "frontier"
    assert models[0].price_in_per_mtok == 15.0
    assert models[0].version_pin == "anthropic/claude-opus-4@2026-xx"
    assert models[1].cls == "open"
    assert models[1].supports_response_format is False
    # Cost prices are plain floats, never Decimal.
    assert isinstance(models[0].price_out_per_mtok, float)


def test_load_models_empty_file_returns_empty_list(tmp_path):
    from spar.eval.models import load_models

    path = tmp_path / "empty.toml"
    path.write_text("", encoding="utf-8")
    assert load_models(path) == []
