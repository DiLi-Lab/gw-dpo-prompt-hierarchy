"""Integration test for the full SFT pipeline using real data libraries.

Runs the aligned, partial, and misaligned builders at small scale against
the library files under this repository's ``data/`` tree. Auto-skips if
data files are missing (e.g. in CI environments where Steps 0 and 2
have not been run).
"""

import re

import pytest
from pathlib import Path

from src.config.paths import PathsConfig

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PATHS = PathsConfig(project_root=PROJECT_ROOT)

libraries_available = (
    PATHS.l0_rules.exists()
    and PATHS.l1_library.exists()
    and PATHS.l4_library.exists()
    and PATHS.alpaca_train.exists()
    and PATHS.dolly_train.exists()
)


@pytest.mark.skipif(
    not libraries_available,
    reason="Real data libraries not available (l0_rules, l1_library, l4_library, splits)",
)
def test_sft_pipeline_small_scale() -> None:
    """Run the full SFT pipeline at small scale and verify output structure."""
    from datasets import load_from_disk

    from src.data.libraries.l0_rules import load_l0_rules
    from src.data.libraries.l1_prompts import load_l1_library
    from src.data.libraries.l4_tool_outputs import load_l4_library
    from src.data.sft.aligned import build_simple_aligned
    from src.data.sft.build_sft_dataset import compute_sft_stats
    from src.data.sft.misaligned import build_misaligned_examples
    from src.data.sft.partial import build_partial_examples

    # Load libraries
    l0_rules = load_l0_rules(PATHS.l0_rules)
    l1_library = load_l1_library(PATHS.l1_library)
    l4_entries = load_l4_library(PATHS.l4_library)
    l4_lookup = {
        (e.source, e.index): {"l4_content": e.l4_content, "generation": e.generation}
        for e in l4_entries
    }

    alpaca_train = load_from_disk(str(PATHS.alpaca_train))
    dolly_train = load_from_disk(str(PATHS.dolly_train))

    # Tag rows with _sft_source and _sft_index
    tagged_rows: list[dict] = []
    for i, row in enumerate(alpaca_train):
        d = dict(row)
        d["_sft_source"] = "alpaca"
        d["_sft_index"] = i
        tagged_rows.append(d)
    for i, row in enumerate(dolly_train):
        d = dict(row)
        d["_sft_source"] = "dolly"
        d["_sft_index"] = i
        tagged_rows.append(d)

    import random
    rng = random.Random(42)
    rng.shuffle(tagged_rows)

    # Pre-filter rows to those with L4 library coverage (aligned builder requires it)
    rows_with_l4 = [r for r in tagged_rows if (r["_sft_source"], r["_sft_index"]) in l4_lookup]
    rows_without_l4 = [r for r in tagged_rows if (r["_sft_source"], r["_sft_index"]) not in l4_lookup]

    # Build 20 simple aligned (must have L4 coverage)
    aligned_count = 20
    aligned = build_simple_aligned(
        base_rows=rows_with_l4[:100],
        l0_rules=l0_rules,
        l1_library=l1_library,
        l4_lookup=l4_lookup,
        count=aligned_count,
        seed=42,
    )
    assert len(aligned) == aligned_count

    # Build 4 configs x 5 partial
    partial_per_config = 5
    partial = build_partial_examples(
        base_rows=tagged_rows[100:200],
        l0_rules=l0_rules,
        l1_library=l1_library,
        l4_lookup=l4_lookup,
        per_config_count=partial_per_config,
        seed=42,
    )
    assert len(partial) == 4 * partial_per_config

    # Build 4 types x 5 misaligned
    misaligned_per_type = 5
    misaligned = build_misaligned_examples(
        l0_rules=l0_rules,
        l1_library=l1_library,
        base_rows=tagged_rows[200:1000],
        per_type_count=misaligned_per_type,
        seed=42,
    )
    assert len(misaligned) == 4 * misaligned_per_type

    # Combine all examples
    all_examples = aligned + partial + misaligned
    expected_total = aligned_count + 4 * partial_per_config + 4 * misaligned_per_type
    assert len(all_examples) == expected_total  # 20 + 20 + 20 = 60

    # Compute stats and verify aligned/conflicting counts
    stats = compute_sft_stats(all_examples)
    assert stats["total"] == expected_total
    assert stats["aligned"] > 0, "Expected some aligned examples"
    assert stats["conflicting"] > 0, "Expected some conflicting examples"

    # Verify delimiter structure for every example
    level_start_re = re.compile(r"<\|L(\d+)_START\|>")
    level_end_re = re.compile(r"<\|L(\d+)_END\|>")

    for i, ex in enumerate(all_examples):
        text = ex["text"]
        levels_present = ex["levels_present"]

        # Every level in levels_present must have matching START and END
        for level in levels_present:
            start_tag = "<|L%d_START|>" % level
            end_tag = "<|L%d_END|>" % level
            assert start_tag in text, (
                "Example %d missing %s (levels_present=%s)" % (i, start_tag, levels_present)
            )
            assert end_tag in text, (
                "Example %d missing %s (levels_present=%s)" % (i, end_tag, levels_present)
            )

        # Every START tag in text must have a matching END tag
        starts = level_start_re.findall(text)
        ends = level_end_re.findall(text)
        assert sorted(starts) == sorted(ends), (
            "Example %d has mismatched START/END tags: starts=%s ends=%s"
            % (i, starts, ends)
        )

        # Every example must have RESP_START and RESP_END
        assert "<|RESP_START|>" in text, "Example %d missing <|RESP_START|>" % i
        assert "<|RESP_END|>" in text, "Example %d missing <|RESP_END|>" % i
