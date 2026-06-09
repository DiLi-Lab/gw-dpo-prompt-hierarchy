"""Configuration system for the 5-level instruction hierarchy project.

Usage:
    from src.config import load_config, ProjectConfig
    cfg = load_config(config_path=Path("configs/base_linear.yaml"))
"""

from src.config.constants import (
    HIERARCHY_LEVELS,
    LEVEL_NAMES,
    LEVEL_TO_SEGMENT_ID,
    LLAMA_MODEL_PATH,
    NUM_LEVELS,
    NUM_SEGMENTS,
    RESPONSE_SEGMENT_ID,
    SPECIAL_TOKENS,
)
from src.config.hierarchy import (
    CONFLICT_LABELS,
    CONFLICT_PAIRS,
    get_level_gap,
    is_valid_conflict,
)
from src.config.hyperparameters import DPOConfig, ModelConfig, SFTConfig
from src.config.loader import ProjectConfig, load_config
from src.config.paths import PathsConfig

__all__ = [
    "CONFLICT_LABELS",
    "CONFLICT_PAIRS",
    "DPOConfig",
    "HIERARCHY_LEVELS",
    "LEVEL_NAMES",
    "LEVEL_TO_SEGMENT_ID",
    "LLAMA_MODEL_PATH",
    "ModelConfig",
    "NUM_LEVELS",
    "NUM_SEGMENTS",
    "PathsConfig",
    "ProjectConfig",
    "RESPONSE_SEGMENT_ID",
    "SFTConfig",
    "SPECIAL_TOKENS",
    "get_level_gap",
    "is_valid_conflict",
    "load_config",
]
