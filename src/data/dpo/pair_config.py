"""DPO pair configuration definitions.

Defines PairConfig dataclass and all 12 DPO pair type configurations
used in the Gravity-Weighted DPO training pipeline.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class PairConfig:
    """Configuration for a single DPO pair type."""

    name: str
    victim_level: int
    attacker_level: int
    target_count: int
    category: str
    yw_strategy: str
    yl_strategy: str
    l2_conflict: bool
    l2_conflict_attribute: str | None
    injection_method: str | None
    injection_target_level: int | None
    needs_summarisation_rows: bool
    phase: int
    scenario_driven: bool = False

    @property
    def level_gap(self) -> int:
        """Distance between attacker and victim levels."""
        return self.attacker_level - self.victim_level

    @property
    def margin(self) -> float:
        """DPO margin derived from level gap."""
        return float(self.level_gap)


ALL_PAIR_CONFIGS: list[PairConfig] = [
    PairConfig(
        name="L0_vs_L1",
        victim_level=0,
        attacker_level=1,
        target_count=500,
        category="pairwise",
        yw_strategy="claude_distillation",
        yl_strategy="gpt4o_mini",
        l2_conflict=False,
        l2_conflict_attribute=None,
        injection_method=None,
        injection_target_level=None,
        needs_summarisation_rows=False,
        phase=3,
        scenario_driven=True,
    ),
    PairConfig(
        name="L0_vs_L2",
        victim_level=0,
        attacker_level=2,
        target_count=500,
        category="pairwise",
        yw_strategy="claude_distillation",
        yl_strategy="gpt4o_mini",
        l2_conflict=True,
        l2_conflict_attribute="content_filter",
        injection_method=None,
        injection_target_level=None,
        needs_summarisation_rows=False,
        phase=3,
        scenario_driven=True,
    ),
    PairConfig(
        name="L0_vs_L3",
        victim_level=0,
        attacker_level=3,
        target_count=500,
        category="pairwise",
        yw_strategy="template",
        yl_strategy="gpt4o_mini",
        l2_conflict=False,
        l2_conflict_attribute=None,
        injection_method="prefix",
        injection_target_level=3,
        needs_summarisation_rows=False,
        phase=2,
    ),
    PairConfig(
        name="L0_vs_L4",
        victim_level=0,
        attacker_level=4,
        target_count=500,
        category="pairwise",
        yw_strategy="claude_distillation",
        yl_strategy="gpt4o_mini",
        l2_conflict=False,
        l2_conflict_attribute=None,
        injection_method="position",
        injection_target_level=4,
        needs_summarisation_rows=False,
        phase=3,
    ),
    PairConfig(
        name="L1_vs_L2",
        victim_level=1,
        attacker_level=2,
        target_count=500,
        category="pairwise",
        yw_strategy="gpt4o_mini_reformat",
        yl_strategy="gpt4o_mini_reformat",
        l2_conflict=True,
        l2_conflict_attribute="format",
        injection_method=None,
        injection_target_level=None,
        needs_summarisation_rows=False,
        phase=2,
    ),
    PairConfig(
        name="L1_vs_L3",
        victim_level=1,
        attacker_level=3,
        target_count=1500,
        category="pairwise",
        yw_strategy="base_dataset",
        yl_strategy="base_dataset",
        l2_conflict=False,
        l2_conflict_attribute=None,
        injection_method="prefix",
        injection_target_level=3,
        needs_summarisation_rows=False,
        phase=1,
    ),
    PairConfig(
        name="L1_vs_L4",
        victim_level=1,
        attacker_level=4,
        target_count=1000,
        category="pairwise",
        yw_strategy="claude_distillation",
        yl_strategy="gpt4o_mini",
        l2_conflict=False,
        l2_conflict_attribute=None,
        injection_method="position",
        injection_target_level=4,
        needs_summarisation_rows=False,
        phase=2,
    ),
    PairConfig(
        name="L2_vs_L3",
        victim_level=2,
        attacker_level=3,
        target_count=500,
        category="pairwise",
        yw_strategy="gpt4o_mini_reformat",
        yl_strategy="base_dataset",
        l2_conflict=True,
        l2_conflict_attribute="format",
        injection_method=None,
        injection_target_level=None,
        needs_summarisation_rows=False,
        phase=2,
    ),
    PairConfig(
        name="L2_vs_L4",
        victim_level=2,
        attacker_level=4,
        target_count=500,
        category="pairwise",
        yw_strategy="gpt4o_mini_reformat",
        yl_strategy="gpt4o_mini_reformat",
        l2_conflict=True,
        l2_conflict_attribute="format",
        injection_method="position",
        injection_target_level=4,
        needs_summarisation_rows=False,
        phase=2,
    ),
    PairConfig(
        name="L3_vs_L4",
        victim_level=3,
        attacker_level=4,
        target_count=1000,
        category="pairwise",
        yw_strategy="base_dataset",
        yl_strategy="gpt4o_mini",
        l2_conflict=False,
        l2_conflict_attribute=None,
        injection_method="position",
        injection_target_level=4,
        needs_summarisation_rows=False,
        phase=2,
    ),
    PairConfig(
        name="calibration",
        victim_level=3,
        attacker_level=3,
        target_count=2000,
        category="calibration",
        yw_strategy="base_dataset",
        yl_strategy="template",
        l2_conflict=False,
        l2_conflict_attribute=None,
        injection_method=None,
        injection_target_level=None,
        needs_summarisation_rows=False,
        phase=2,
    ),
    PairConfig(
        name="cascading",
        victim_level=0,
        attacker_level=4,
        target_count=1000,
        category="cascading",
        yw_strategy="claude_distillation",
        yl_strategy="gpt4o_mini",
        l2_conflict=False,
        l2_conflict_attribute=None,
        injection_method=None,
        injection_target_level=None,
        needs_summarisation_rows=False,
        phase=3,
    ),
]

# Validation pair configs — same structure, scaled to ~1,000 total
_VAL_TARGET_COUNTS: dict[str, int] = {
    "L0_vs_L1": 50,
    "L0_vs_L2": 50,
    "L0_vs_L3": 50,
    "L0_vs_L4": 50,
    "L1_vs_L2": 50,
    "L1_vs_L3": 150,
    "L1_vs_L4": 100,
    "L2_vs_L3": 50,
    "L2_vs_L4": 50,
    "L3_vs_L4": 100,
    "calibration": 200,
    "cascading": 100,
}

VAL_PAIR_CONFIGS: list[PairConfig] = [
    PairConfig(
        name=c.name,
        victim_level=c.victim_level,
        attacker_level=c.attacker_level,
        target_count=_VAL_TARGET_COUNTS[c.name],
        category=c.category,
        yw_strategy=c.yw_strategy,
        yl_strategy=c.yl_strategy,
        l2_conflict=c.l2_conflict,
        l2_conflict_attribute=c.l2_conflict_attribute,
        injection_method=c.injection_method,
        injection_target_level=c.injection_target_level,
        needs_summarisation_rows=c.needs_summarisation_rows,
        phase=c.phase,
        scenario_driven=c.scenario_driven,
    )
    for c in ALL_PAIR_CONFIGS
]


def get_pair_configs(split: str | None = None) -> list[PairConfig]:
    """Return the pair config list for the given split.

    Args:
        split: "train", "val", or None. None and "train" return ALL_PAIR_CONFIGS.

    Returns:
        The appropriate list of PairConfig objects.
    """
    if split == "val":
        return VAL_PAIR_CONFIGS
    return ALL_PAIR_CONFIGS


_CONFIG_BY_NAME: dict[str, PairConfig] = {c.name: c for c in ALL_PAIR_CONFIGS}


def get_config_by_name(name: str) -> PairConfig:
    """Look up a PairConfig by name.

    Args:
        name: The configuration name (e.g. "L0_vs_L1", "calibration").

    Returns:
        The matching PairConfig.

    Raises:
        KeyError: If no configuration matches the given name.
    """
    return _CONFIG_BY_NAME[name]
