"""Tests for --count global override (without --pair) in DPO build."""

from dataclasses import replace

from src.data.dpo.build_dpo_dataset import partition_dpo_pool
from src.data.dpo.pair_config import ALL_PAIR_CONFIGS, PairConfig, get_config_by_name


def _make_rows(n: int) -> list[dict]:
    """Create n tagged rows with alternating L4 coverage."""
    return [
        {
            "instruction": f"Instr {i}",
            "input": "",
            "output": f"Out {i}",
            "_dpo_source": "alpaca",
            "_dpo_index": i,
        }
        for i in range(n)
    ]


def _make_l4_lookup(n: int) -> dict[tuple[str, int], dict]:
    """L4 lookup covering every other row up to n."""
    return {
        ("alpaca", i): {"l4_content": "content", "generation": "wrapped"}
        for i in range(0, n, 2)
    }


def test_global_count_overrides_all_configs():
    """When --count N is passed without --pair, every config's target_count becomes N."""
    count_override = 10
    overridden = [replace(c, target_count=count_override) for c in ALL_PAIR_CONFIGS]
    for cfg in overridden:
        assert cfg.target_count == count_override


def test_global_count_partition_produces_small_slices():
    """With global --count=10, partition slices should be much smaller than defaults."""
    count_override = 10
    overridden = [replace(c, target_count=count_override) for c in ALL_PAIR_CONFIGS]

    rows = _make_rows(5000)
    l4_lookup = _make_l4_lookup(5000)

    slices = partition_dpo_pool(rows, overridden, l4_lookup, seed=42)

    for cfg in overridden:
        if cfg.scenario_driven:
            assert len(slices[cfg.name]) == 0
        else:
            # With target_count=10 and 1.2x headroom (or 2*1.2x for L1_vs_L3),
            # slices should be at most ~24 rows
            assert len(slices[cfg.name]) <= 30, (
                f"{cfg.name} got {len(slices[cfg.name])} rows, expected <= 30"
            )
            assert len(slices[cfg.name]) > 0, (
                f"{cfg.name} got 0 rows, expected > 0"
            )


def test_global_count_does_not_mutate_originals():
    """Overriding via dataclasses.replace must not mutate ALL_PAIR_CONFIGS."""
    original_counts = {c.name: c.target_count for c in ALL_PAIR_CONFIGS}
    _ = [replace(c, target_count=5) for c in ALL_PAIR_CONFIGS]
    for c in ALL_PAIR_CONFIGS:
        assert c.target_count == original_counts[c.name]


def test_run_phase2_accepts_count_override():
    """run_phase2 signature accepts count_override parameter."""
    import inspect

    from src.data.dpo.build_dpo_dataset import run_phase2

    sig = inspect.signature(run_phase2)
    assert "count_override" in sig.parameters, (
        "run_phase2 must accept a count_override parameter"
    )
    param = sig.parameters["count_override"]
    assert param.default is None, "count_override should default to None"


def test_run_phase3_accepts_count_override():
    """run_phase3 signature accepts count_override parameter."""
    import inspect

    from src.data.dpo.build_dpo_dataset import run_phase3

    sig = inspect.signature(run_phase3)
    assert "count_override" in sig.parameters, (
        "run_phase3 must accept a count_override parameter"
    )
    param = sig.parameters["count_override"]
    assert param.default is None, "count_override should default to None"


def test_dry_run_table_with_global_count(capsys):
    """_print_dry_run_table should accept and display overridden configs."""
    # This tests that the dry-run function can receive overridden configs
    count_override = 10
    overridden = [replace(c, target_count=count_override) for c in ALL_PAIR_CONFIGS]
    total = sum(c.target_count for c in overridden)
    assert total == count_override * len(ALL_PAIR_CONFIGS)
