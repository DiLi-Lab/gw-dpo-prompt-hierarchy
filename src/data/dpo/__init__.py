"""DPO dataset construction pipeline."""

from src.data.dpo.assembly import assemble_dpo_example, assemble_dpo_prompt
from src.data.dpo.pair_config import ALL_PAIR_CONFIGS, PairConfig, get_config_by_name

__all__ = [
    "PairConfig",
    "ALL_PAIR_CONFIGS",
    "get_config_by_name",
    "assemble_dpo_prompt",
    "assemble_dpo_example",
]
