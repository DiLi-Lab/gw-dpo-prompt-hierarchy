"""Model nickname → checkpoint resolution."""

from pathlib import Path

import pytest

from src.config.paths import PathsConfig
from src.evaluation.external.registry import (
    KNOWN_MODELS,
    ResolvedModel,
    resolve_model,
)


def test_base_stock_does_not_require_special_tokens() -> None:
    paths = PathsConfig(project_root=Path("/tmp/pp"))
    resolved = resolve_model("base_stock", paths)
    assert isinstance(resolved, ResolvedModel)
    assert resolved.model_path == "meta-llama/Llama-3.1-8B-Instruct"
    assert resolved.requires_special_tokens is False
    assert resolved.kind == "base"


def test_gw_dpo_resolves_to_local_dir() -> None:
    paths = PathsConfig(project_root=Path("/tmp/pp"))
    resolved = resolve_model("gw_dpo", paths)
    assert str(resolved.model_path).endswith("llama-3.1-8b-gw-dpo-final")
    assert resolved.requires_special_tokens is True
    assert resolved.kind == "trained"


def test_tokens_only_uses_no_ise_dirname() -> None:
    paths = PathsConfig(project_root=Path("/tmp/pp"))
    resolved = resolve_model("tokens_only", paths)
    assert str(resolved.model_path).endswith("llama-3.1-8b-gw-dpo-no-ise-final")


def test_unknown_nickname_lists_known_options() -> None:
    paths = PathsConfig(project_root=Path("/tmp/pp"))
    with pytest.raises(ValueError) as excinfo:
        resolve_model("typo_dpo", paths)
    msg = str(excinfo.value)
    for nick in ("base_stock", "gw_dpo", "bilateral", "three_level"):
        assert nick in msg


def test_known_models_table_is_exposed() -> None:
    assert "gw_dpo" in KNOWN_MODELS
    assert "base_stock" in KNOWN_MODELS
    assert len(KNOWN_MODELS) == 8
