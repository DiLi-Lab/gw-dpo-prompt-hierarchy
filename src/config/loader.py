"""Config loader: merges defaults, YAML overrides, and CLI overrides.

Priority order (highest wins): CLI overrides > YAML file > dataclass defaults.
"""

import dataclasses
from dataclasses import fields
from pathlib import Path

import yaml

from src.config.hyperparameters import (
    DPOConfig,
    EvalConfig,
    EvaluationConfig,
    IHEvalConfig,
    ModelConfig,
    MTBenchConfig,
    SEPConfig,
    SFTConfig,
    TensorTrustConfig,
    XSTestConfig,
)

# Fields that should be converted from YAML lists to tuples
_TUPLE_FIELDS = {
    "lora_target_modules",
    "default_tasks",
    "default_settings",
    "scoring_refusal_patterns",
}
from src.config.paths import PathsConfig


@dataclasses.dataclass
class ProjectConfig:
    """Top-level config aggregating all sub-configs."""

    model: ModelConfig = dataclasses.field(default_factory=ModelConfig)
    paths: PathsConfig = dataclasses.field(default_factory=PathsConfig)
    sft: SFTConfig = dataclasses.field(default_factory=SFTConfig)
    dpo: DPOConfig = dataclasses.field(default_factory=DPOConfig)
    eval: EvalConfig = dataclasses.field(default_factory=EvalConfig)
    evaluation: EvaluationConfig = dataclasses.field(default_factory=EvaluationConfig)
    xstest: XSTestConfig | None = None
    iheval: IHEvalConfig | None = None
    sep: SEPConfig | None = None
    mt_bench: MTBenchConfig | None = None
    tensortrust: TensorTrustConfig | None = None


_SECTION_MAP: dict[str, type] = {
    "model": ModelConfig,
    "paths": PathsConfig,
    "sft": SFTConfig,
    "dpo": DPOConfig,
    "eval": EvalConfig,
    "evaluation": EvaluationConfig,
    "xstest": XSTestConfig,
    "iheval": IHEvalConfig,
    "sep": SEPConfig,
    "mt_bench": MTBenchConfig,
    "tensortrust": TensorTrustConfig,
}


def _coerce_value(dataclass_type: type, field_name: str, raw_value: str) -> object:
    """Coerce a string value to the correct type for a dataclass field."""
    field_types = {f.name: f.type for f in fields(dataclass_type)}
    if field_name not in field_types:
        msg = f"Unknown field '{field_name}' in {dataclass_type.__name__}"
        raise ValueError(msg)

    target_type = field_types[field_name]
    if target_type in (int, "int"):
        return int(raw_value)
    if target_type in (float, "float"):
        return float(raw_value)
    if target_type in (bool, "bool"):
        return raw_value.lower() in ("true", "1", "yes")
    if target_type in (str, "str"):
        return raw_value
    if target_type in (Path, "Path"):
        return Path(raw_value)
    return raw_value


_BASE_CONFIG_PATH = Path("configs/base_linear.yaml")


def _load_yaml_into(section_dicts: dict[str, dict], yaml_path: Path) -> None:
    """Merge all known sections from yaml_path into section_dicts (in place)."""
    with open(yaml_path) as f:
        yaml_data = yaml.safe_load(f) or {}
    for section_name, section_values in yaml_data.items():
        if section_name in section_dicts and isinstance(section_values, dict):
            section_dicts[section_name].update(section_values)


def load_config(
    config_path: Path | None = None,
    overrides: list[str] | None = None,
    base_config_path: Path | None = _BASE_CONFIG_PATH,
) -> ProjectConfig:
    """Load project configuration with YAML and CLI overrides.

    Args:
        config_path: Path to a YAML config file. If None, uses defaults.
        overrides: List of "section.key=value" strings for CLI overrides.
        base_config_path: Path to base config loaded first as defaults.
            When non-None and the file exists, it is loaded before
            config_path so that standalone benchmark configs (which only
            carry an xstest: or iheval: section) do not need to repeat
            model/sft/dpo/eval/evaluation sections. Defaults to
            configs/base_linear.yaml. Pass None to disable chain-loading.

    Returns:
        Fully resolved ProjectConfig instance.

    Raises:
        FileNotFoundError: If config_path does not exist.
        ValueError: If an override references an unknown section or field.
    """
    section_dicts: dict[str, dict] = {name: {} for name in _SECTION_MAP}

    # Chain-load base config first (provides defaults for model/sft/dpo/etc.)
    # Only chain when the caller provided a config_path that differs from base;
    # this preserves the original behaviour of load_config() with no args
    # (which raises TypeError due to missing required fields).
    if (
        config_path is not None
        and base_config_path is not None
        and base_config_path.exists()
        and config_path.resolve() != base_config_path.resolve()
    ):
        _load_yaml_into(section_dicts, base_config_path)

    # Layer the user's config on top of base defaults
    if config_path is not None:
        if not config_path.exists():
            msg = f"Config file not found: {config_path}"
            raise FileNotFoundError(msg)
        _load_yaml_into(section_dicts, config_path)

    if overrides:
        for override in overrides:
            key, _, value = override.partition("=")
            section_name, _, field_name = key.partition(".")
            if section_name not in _SECTION_MAP:
                msg = f"Unknown config section: '{section_name}'"
                raise ValueError(msg)
            coerced = _coerce_value(_SECTION_MAP[section_name], field_name, value)
            section_dicts[section_name][field_name] = coerced

    # Convert YAML lists to tuples for fields that expect tuples
    for section_name in ("sft", "dpo", "iheval", "sep"):
        for field_name in _TUPLE_FIELDS:
            if field_name in section_dicts[section_name]:
                val = section_dicts[section_name][field_name]
                if isinstance(val, list):
                    section_dicts[section_name][field_name] = tuple(val)

    return ProjectConfig(
        model=ModelConfig(**section_dicts["model"]),
        paths=PathsConfig(**{
            k: Path(v) if k == "project_root" else v
            for k, v in section_dicts["paths"].items()
        }),
        sft=SFTConfig(**section_dicts["sft"]),
        dpo=DPOConfig(**section_dicts["dpo"]),
        eval=EvalConfig(**section_dicts["eval"]),
        evaluation=EvaluationConfig(**section_dicts["evaluation"]),
        xstest=XSTestConfig(**section_dicts["xstest"]) if section_dicts["xstest"] else None,
        iheval=IHEvalConfig(**section_dicts["iheval"]) if section_dicts["iheval"] else None,
        sep=SEPConfig(**section_dicts["sep"]) if section_dicts["sep"] else None,
        mt_bench=MTBenchConfig(**section_dicts["mt_bench"]) if section_dicts["mt_bench"] else None,
        tensortrust=(
            TensorTrustConfig(**section_dicts["tensortrust"])
            if section_dicts["tensortrust"]
            else None
        ),
    )
