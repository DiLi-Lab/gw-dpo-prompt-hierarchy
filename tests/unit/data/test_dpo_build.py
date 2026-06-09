"""Tests for DPO build utilities."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.data.dpo.build_dpo_dataset import (
    compute_dpo_stats,
    exclude_prior_phase_rows,
    save_dpo_cache,
    load_dpo_cache,
    exclude_sft_rows,
    partition_dpo_pool,
    run_phase1,
    run_phase2,
    run_phase3,
    combine_phases,
)
from src.data.dpo.pair_config import ALL_PAIR_CONFIGS


def _make_dpo_example(conflict_type="L1_vs_L3", category="pairwise",
                      yw_source="base_dataset", yl_source="base_dataset",
                      yl_fallback_used=None) -> dict:
    return {
        "prompt": "...",
        "chosen": "<|RESP_START|>good<|RESP_END|>",
        "rejected": "<|RESP_START|>bad<|RESP_END|>",
        "conflict_type": conflict_type,
        "category": category,
        "yw_source": yw_source,
        "yl_source": yl_source,
        "yl_fallback_used": yl_fallback_used,
    }


def test_compute_dpo_stats_basic():
    examples = [
        _make_dpo_example("L1_vs_L3"),
        _make_dpo_example("L1_vs_L3"),
        _make_dpo_example("L0_vs_L4", category="pairwise", yl_source="gpt4o_mini"),
        _make_dpo_example("calibration_L3", category="calibration"),
    ]
    stats = compute_dpo_stats(examples)
    assert stats["total"] == 4
    assert stats["conflict_types"]["L1_vs_L3"] == 2
    assert stats["conflict_types"]["L0_vs_L4"] == 1
    assert stats["categories"]["pairwise"] == 3
    assert stats["categories"]["calibration"] == 1
    assert stats["yw_sources"]["base_dataset"] == 4
    assert stats["yl_sources"]["base_dataset"] == 3
    assert stats["yl_sources"]["gpt4o_mini"] == 1


def test_compute_dpo_stats_fallback_tracking():
    examples = [
        _make_dpo_example(yl_fallback_used=None),
        _make_dpo_example(yl_fallback_used="rephrase"),
        _make_dpo_example(yl_fallback_used="string_concat"),
    ]
    stats = compute_dpo_stats(examples)
    assert stats["yl_fallbacks"][None] == 1
    assert stats["yl_fallbacks"]["rephrase"] == 1
    assert stats["yl_fallbacks"]["string_concat"] == 1


def test_save_load_cache_roundtrip(tmp_path):
    cache = {
        ("L1_vs_L3", "alpaca", 0): "cached response 0",
        ("L0_vs_L4", "dolly", 5): "cached response 5",
    }
    path = tmp_path / "test_cache.jsonl"
    save_dpo_cache(cache, path)
    loaded = load_dpo_cache(path)
    assert loaded == cache


def test_load_cache_missing_file(tmp_path):
    path = tmp_path / "missing.jsonl"
    loaded = load_dpo_cache(path)
    assert loaded == {}


def test_exclude_sft_rows(tmp_path):
    # Create fake SFT file
    sft_path = tmp_path / "sft_combined.jsonl"
    sft_rows = [
        {"sft_source": "alpaca", "sft_index": 0, "text": "..."},
        {"sft_source": "alpaca", "sft_index": 5, "text": "..."},
        {"sft_source": "dolly", "sft_index": 10, "text": "..."},
    ]
    with open(sft_path, "w") as f:
        for row in sft_rows:
            f.write(json.dumps(row) + "\n")

    all_rows = [
        {"instruction": "a", "_dpo_source": "alpaca", "_dpo_index": 0},  # SFT used
        {"instruction": "b", "_dpo_source": "alpaca", "_dpo_index": 1},  # available
        {"instruction": "c", "_dpo_source": "alpaca", "_dpo_index": 5},  # SFT used
        {"instruction": "d", "_dpo_source": "dolly", "_dpo_index": 10},  # SFT used
        {"instruction": "e", "_dpo_source": "dolly", "_dpo_index": 11},  # available
    ]
    remaining = exclude_sft_rows(all_rows, sft_path)
    assert len(remaining) == 2
    indices = [(r["_dpo_source"], r["_dpo_index"]) for r in remaining]
    assert ("alpaca", 1) in indices
    assert ("dolly", 11) in indices


def test_exclude_prior_phase_rows_filters_used_instances(tmp_path):
    """exclude_prior_phase_rows removes pool rows whose (source, index) appear
    as yw_base or yl_base in any existing phase output file."""
    # Create a fake phase output with known base instances
    phase1_path = tmp_path / "phase1.jsonl"
    phase1_examples = [
        {
            "yw_base_dataset": "alpaca", "yw_base_index": 0,
            "yl_base_dataset": "alpaca", "yl_base_index": 1,
            "prompt": "a", "chosen": "c", "rejected": "r",
        },
        {
            "yw_base_dataset": "dolly", "yw_base_index": 5,
            "yl_base_dataset": "dolly", "yl_base_index": 6,
            "prompt": "b", "chosen": "c", "rejected": "r",
        },
    ]
    with open(phase1_path, "w") as f:
        for ex in phase1_examples:
            f.write(json.dumps(ex) + "\n")

    pool = [
        {"instruction": "a", "_dpo_source": "alpaca", "_dpo_index": 0},   # used as yw
        {"instruction": "b", "_dpo_source": "alpaca", "_dpo_index": 1},   # used as yl
        {"instruction": "c", "_dpo_source": "alpaca", "_dpo_index": 2},   # free
        {"instruction": "d", "_dpo_source": "dolly", "_dpo_index": 5},    # used as yw
        {"instruction": "e", "_dpo_source": "dolly", "_dpo_index": 6},    # used as yl
        {"instruction": "f", "_dpo_source": "dolly", "_dpo_index": 7},    # free
    ]
    remaining = exclude_prior_phase_rows(pool, [phase1_path])
    assert len(remaining) == 2
    keys = {(r["_dpo_source"], r["_dpo_index"]) for r in remaining}
    assert keys == {("alpaca", 2), ("dolly", 7)}


def test_exclude_prior_phase_rows_skips_missing_files(tmp_path):
    """Missing phase files are silently skipped."""
    pool = [
        {"instruction": "a", "_dpo_source": "alpaca", "_dpo_index": 0},
    ]
    remaining = exclude_prior_phase_rows(
        pool, [tmp_path / "nonexistent.jsonl"],
    )
    assert len(remaining) == 1


def test_exclude_prior_phase_rows_handles_multiple_files(tmp_path):
    """Rows used across multiple prior phases are all excluded."""
    p1 = tmp_path / "p1.jsonl"
    p2 = tmp_path / "p2.jsonl"
    with open(p1, "w") as f:
        f.write(json.dumps({
            "yw_base_dataset": "alpaca", "yw_base_index": 0,
            "yl_base_dataset": "alpaca", "yl_base_index": 1,
        }) + "\n")
    with open(p2, "w") as f:
        f.write(json.dumps({
            "yw_base_dataset": "alpaca", "yw_base_index": 2,
            "yl_base_dataset": "alpaca", "yl_base_index": 3,
        }) + "\n")

    pool = [
        {"instruction": str(i), "_dpo_source": "alpaca", "_dpo_index": i}
        for i in range(5)
    ]
    remaining = exclude_prior_phase_rows(pool, [p1, p2])
    assert len(remaining) == 1
    assert remaining[0]["_dpo_index"] == 4


def test_exclude_prior_phase_rows_handles_null_base_fields(tmp_path):
    """Examples with null yw/yl base fields (e.g. scenario-driven) are ignored."""
    p1 = tmp_path / "p1.jsonl"
    with open(p1, "w") as f:
        f.write(json.dumps({
            "yw_base_dataset": None, "yw_base_index": None,
            "yl_base_dataset": None, "yl_base_index": None,
        }) + "\n")
        f.write(json.dumps({
            "yw_base_dataset": "alpaca", "yw_base_index": 10,
            "yl_base_dataset": "alpaca", "yl_base_index": 11,
        }) + "\n")

    pool = [
        {"instruction": str(i), "_dpo_source": "alpaca", "_dpo_index": i}
        for i in range(12)
    ]
    remaining = exclude_prior_phase_rows(pool, [p1])
    assert len(remaining) == 10  # only indices 10 and 11 excluded


def test_partition_dpo_pool():
    # Create a large pool to accommodate all configs (total target ~20k with headroom)
    pool_size = 30000
    rows = []
    for i in range(pool_size):
        # Every 5th row has "summarise" in instruction for summarisation filter
        instr = f"Please summarise document {i}" if i % 5 == 0 else f"Instruction {i}"
        rows.append({
            "instruction": instr,
            "input": "data" if i % 3 == 0 else "",
            "output": f"Output {i}",
            "_dpo_source": "alpaca",
            "_dpo_index": i,
        })
    # L4 lookup covers every 2nd row
    l4_lookup = {
        ("alpaca", i): {"l4_content": "content", "generation": "wrapped"}
        for i in range(0, pool_size, 2)
    }
    slices = partition_dpo_pool(rows, ALL_PAIR_CONFIGS, l4_lookup, seed=42)
    assert isinstance(slices, dict)
    # Should have a key for each config name
    for cfg in ALL_PAIR_CONFIGS:
        assert cfg.name in slices, f"Missing slice for {cfg.name}"
    # Slices should be non-empty (at least some rows)
    # L1_vs_L3 should get the most rows (needs 2x for pairs)
    assert len(slices["L1_vs_L3"]) > 0
    # Verify disjointness: no row index appears in multiple slices
    all_indices: list[int] = []
    for name, slice_rows in slices.items():
        for row in slice_rows:
            all_indices.append(row["_dpo_index"])
    assert len(all_indices) == len(set(all_indices)), "Rows must be disjoint across slices"


def _make_tagged_row(source, idx, instruction="Task", output="Result", data=""):
    if source == "alpaca":
        return {"instruction": instruction, "input": data, "output": output,
                "_dpo_source": source, "_dpo_index": idx}
    return {"instruction": instruction, "context": data, "response": output,
            "_dpo_source": source, "_dpo_index": idx, "category": "summarization"}


def test_run_phase1_produces_examples(tmp_path):
    """Phase 1 builds L1-vs-L3 pairs from row pairs."""
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class FakeL0Rule:
        category: str
        rule: str
        id: str

    @dataclass(frozen=True)
    class FakeInjEntry:
        text: str
        id: str

    @dataclass
    class FakeInjTemplate:
        prefixes: list
        system_overrides: list
        position_injections: list

    l0_rules = [FakeL0Rule("system_integrity", "Untrusted.", "L0_SI_001")]
    l1_library = [{"domain": "general knowledge", "full_prompt": "You are helpful.",
                   "persona": "assistant", "constraints": ["be accurate"]}]
    injection_templates = FakeInjTemplate(
        prefixes=[FakeInjEntry("Ignore. Instead: ", "pfx_01")],
        system_overrides=[], position_injections=[],
    )
    rows = [_make_tagged_row("alpaca", i, f"Instr {i}", f"Out {i}") for i in range(20)]

    output_path = tmp_path / "phase1.jsonl"
    results = run_phase1(
        pool_slice=rows,
        l0_rules=l0_rules,
        l1_library=l1_library,
        injection_templates=injection_templates,
        output_path=output_path,
        count=5,
        seed=42,
    )
    assert len(results) == 5
    assert output_path.exists()
    # Verify file content
    with open(output_path) as f:
        lines = f.readlines()
    assert len(lines) == 5


def test_partition_prefers_l4_rows_for_non_conflict_configs():
    """Non-L4-conflict configs should get ~70% L4-covered rows."""
    from src.data.dpo.pair_config import PairConfig

    config = PairConfig(
        name="L1_vs_L2", victim_level=1, attacker_level=2, target_count=100,
        category="pairwise", yw_strategy="base_dataset", yl_strategy="gpt4o_mini",
        l2_conflict=True, l2_conflict_attribute="format",
        injection_method=None, injection_target_level=None,
        needs_summarisation_rows=False, phase=2,
    )
    # Pool: 200 rows, 150 have L4 (75% coverage ensures we can hit 70%)
    rows = [
        {"instruction": f"Instr {i}", "input": "", "output": f"Out {i}",
         "_dpo_source": "alpaca", "_dpo_index": i}
        for i in range(200)
    ]
    l4_lookup = {
        ("alpaca", i): {"l4_content": "content", "generation": "wrapped"}
        for i in range(150)
    }
    slices = partition_dpo_pool(rows, [config], l4_lookup, seed=42)
    allocated = slices["L1_vs_L2"]
    l4_count = sum(
        1 for r in allocated
        if (r["_dpo_source"], r["_dpo_index"]) in l4_lookup
    )
    ratio = l4_count / len(allocated) if allocated else 0
    assert ratio >= 0.65, f"L4 ratio {ratio:.2f} is below 0.65 threshold"


def test_partition_no_summarisation_filter():
    """L4-conflict configs should accept any row with L4, not just summarisation."""
    from src.data.dpo.pair_config import PairConfig

    config = PairConfig(
        name="L3_vs_L4", victim_level=3, attacker_level=4, target_count=10,
        category="pairwise", yw_strategy="base_dataset", yl_strategy="gpt4o_mini",
        l2_conflict=False, l2_conflict_attribute=None,
        injection_method="position", injection_target_level=4,
        needs_summarisation_rows=False, phase=2,
    )
    # No rows contain "summar" in instruction — old code would reject all
    rows = [
        {"instruction": f"Explain topic {i}", "input": "", "output": f"Out {i}",
         "_dpo_source": "alpaca", "_dpo_index": i}
        for i in range(50)
    ]
    l4_lookup = {
        ("alpaca", i): {"l4_content": "content", "generation": "wrapped"}
        for i in range(50)
    }
    slices = partition_dpo_pool(rows, [config], l4_lookup, seed=42)
    assert len(slices["L3_vs_L4"]) >= 10, "Should allocate rows without summarisation filter"


def test_partition_skips_scenario_driven_configs():
    """Scenario-driven configs should get empty slices."""
    from src.data.dpo.pair_config import PairConfig

    configs = [
        PairConfig(
            name="L0_vs_L1", victim_level=0, attacker_level=1, target_count=500,
            category="pairwise", yw_strategy="claude_distillation", yl_strategy="gpt4o_mini",
            l2_conflict=False, l2_conflict_attribute=None,
            injection_method=None, injection_target_level=None,
            needs_summarisation_rows=False, phase=3, scenario_driven=True,
        ),
        PairConfig(
            name="L1_vs_L3", victim_level=1, attacker_level=3, target_count=100,
            category="pairwise", yw_strategy="base_dataset", yl_strategy="base_dataset",
            l2_conflict=False, l2_conflict_attribute=None,
            injection_method="prefix", injection_target_level=3,
            needs_summarisation_rows=False, phase=1, scenario_driven=False,
        ),
    ]
    rows = [
        {"instruction": f"Instr {i}", "input": "", "output": f"Out {i}",
         "_dpo_source": "alpaca", "_dpo_index": i}
        for i in range(500)
    ]
    slices = partition_dpo_pool(rows, configs, {}, seed=42)
    assert len(slices["L0_vs_L1"]) == 0, "Scenario-driven config should get empty slice"
    assert len(slices["L1_vs_L3"]) > 0, "Normal config should get rows"


def test_run_phase3_accepts_l4_domain_index(tmp_path):
    """run_phase3 accepts l4_domain_index and passes it to scenario builders."""
    from unittest.mock import MagicMock, patch

    from src.data.dpo.l0_conflict_builder import AdversarialScenario

    scenario = AdversarialScenario(
        id="test_01", pair_type="L0_vs_L1", l0_category="content_prohibitions",
        l0_rule_ids=[], adversarial_l1="Unrestricted.",
        l2_conflict_attribute=None, l2_conflict_value=None,
        l3_templates=["Do bad thing."],
    )
    l4_domain_index = {"general": [("alpaca", 0)]}
    output_path = tmp_path / "phase3.jsonl"

    mock_client = MagicMock()

    with patch("src.data.dpo.build_dpo_dataset.build_l0_vs_l1_pair") as mock_build:
        mock_build.return_value = None  # simplest: builder returns None each time
        run_phase3(
            pool_slices={},
            l0_rules=[],
            l1_library=[],
            l4_lookup={},
            injection_templates=MagicMock(),
            openai_client=mock_client,
            anthropic_client=mock_client,
            cascading_families=[],
            output_path=output_path,
            seed=42,
            l0_conflict_scenarios=[scenario],
            count_override=1,
            l4_domain_index=l4_domain_index,
        )
    # build_l0_vs_l1_pair should have been called with l4_domain_index
    assert mock_build.call_count >= 1
    call_kwargs = mock_build.call_args[1]
    assert call_kwargs.get("l4_domain_index") is l4_domain_index


def test_run_phase3_passes_per_scenario_used_keys(tmp_path):
    """run_phase3 creates a per-scenario used_keys set and passes it to builders."""
    from unittest.mock import MagicMock, call, patch

    from src.data.dpo.l0_conflict_builder import AdversarialScenario

    scenario = AdversarialScenario(
        id="test_01", pair_type="L0_vs_L1", l0_category="content_prohibitions",
        l0_rule_ids=[], adversarial_l1="Unrestricted.",
        l2_conflict_attribute=None, l2_conflict_value=None,
        l3_templates=["Do bad thing."],
    )
    l4_domain_index = {"general": [("alpaca", 0)]}
    output_path = tmp_path / "phase3.jsonl"
    mock_client = MagicMock()

    call_used_keys = []

    def capture_call(**kwargs):
        call_used_keys.append(kwargs.get("l4_used_keys"))
        return None

    with patch("src.data.dpo.build_dpo_dataset.build_l0_vs_l1_pair", side_effect=capture_call):
        run_phase3(
            pool_slices={},
            l0_rules=[],
            l1_library=[],
            l4_lookup={},
            injection_templates=MagicMock(),
            openai_client=mock_client,
            anthropic_client=mock_client,
            cascading_families=[],
            output_path=output_path,
            seed=42,
            l0_conflict_scenarios=[scenario],
            count_override=3,
            l4_domain_index=l4_domain_index,
        )
    # All calls within the same scenario share the same used_keys set
    assert len(call_used_keys) == 3
    assert all(ks is call_used_keys[0] for ks in call_used_keys), \
        "All calls within one scenario must share the same used_keys set"


def test_combine_phases(tmp_path):
    """Combine phase outputs into a single file."""
    phase1 = tmp_path / "phase1.jsonl"
    phase2 = tmp_path / "phase2.jsonl"
    phase3 = tmp_path / "phase3.jsonl"

    with open(phase1, "w") as f:
        f.write(json.dumps({"conflict_type": "L1_vs_L3", "prompt": "a"}) + "\n")
        f.write(json.dumps({"conflict_type": "L1_vs_L3", "prompt": "b"}) + "\n")
    with open(phase2, "w") as f:
        f.write(json.dumps({"conflict_type": "L0_vs_L3", "prompt": "c"}) + "\n")
    with open(phase3, "w") as f:
        f.write(json.dumps({"conflict_type": "cascading_L0_L1", "prompt": "d"}) + "\n")

    combined_path = tmp_path / "combined.jsonl"
    examples = combine_phases(phase1, phase2, phase3, combined_path)
    assert len(examples) == 4
    assert combined_path.exists()
    with open(combined_path) as f:
        assert len(f.readlines()) == 4


# ---------------------------------------------------------------
# Phase 5: dual-judge evaluation (separated from Phase 4)
# ---------------------------------------------------------------

def test_run_phase5_judges_stratified_sample(tmp_path):
    """Phase 5 runs dual-judge evaluation on a stratified sample of combined examples."""
    from src.data.dpo.build_dpo_dataset import run_phase5

    # Create a combined JSONL with enough examples for sampling
    combined_path = tmp_path / "dpo_combined.jsonl"
    examples = []
    for i in range(20):
        ex = {
            "prompt": f"<|L0_START|>rule<|L0_END|>\n<|L1_START|>sys<|L1_END|>\n<|L3_START|>msg {i}<|L3_END|>",
            "chosen": f"<|RESP_START|>{'good ' * 15}<|RESP_END|>",
            "rejected": f"<|RESP_START|>{'bad ' * 15}<|RESP_END|>",
            "conflict_type": "L1_vs_L3",
            "category": "pairwise",
        }
        examples.append(ex)
    with open(combined_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    # Mock both judge clients
    gpt_response = json.dumps({"q1": 5, "q2": 4, "q3": 5, "q4": 4, "keep": True, "reason": "ok"})
    gemini_response = json.dumps({"q1": 4, "q2": 5, "q3": 4, "q4": 3, "keep": True, "reason": "ok"})

    mock_openai = MagicMock()
    mock_openai.generate.return_value = gpt_response
    mock_google = MagicMock()
    mock_google.generate.return_value = gemini_response

    qc_results_path = tmp_path / "qc_results.jsonl"
    flagged_path = tmp_path / "flagged.jsonl"

    results = run_phase5(
        combined_path=combined_path,
        openai_client=mock_openai,
        google_client=mock_google,
        qc_results_path=qc_results_path,
        flagged_path=flagged_path,
        seed=42,
    )

    # Should have called both judges for each sampled example
    assert mock_openai.generate.call_count > 0
    assert mock_google.generate.call_count > 0
    assert mock_openai.generate.call_count == mock_google.generate.call_count
    # QC results should be saved
    assert qc_results_path.exists()
    # Results dict should report counts
    assert "sampled" in results
    assert "kept" in results
    assert "discarded" in results
    assert "flagged" in results


def test_run_phase5_saves_flagged_examples(tmp_path):
    """Phase 5 saves disagreement examples to a separate flagged file."""
    from src.data.dpo.build_dpo_dataset import run_phase5

    combined_path = tmp_path / "dpo_combined.jsonl"
    ex = {
        "prompt": "<|L0_START|>rule<|L0_END|>\n<|L3_START|>msg<|L3_END|>",
        "chosen": "<|RESP_START|>" + "good " * 15 + "<|RESP_END|>",
        "rejected": "<|RESP_START|>" + "bad " * 15 + "<|RESP_END|>",
        "conflict_type": "L0_vs_L3",
        "category": "pairwise",
    }
    with open(combined_path, "w") as f:
        for i in range(10):
            f.write(json.dumps({**ex, "prompt": ex["prompt"] + f" {i}"}) + "\n")

    # GPT says keep, Gemini says reject -> flag
    gpt_response = json.dumps({"q1": 5, "q2": 4, "q3": 5, "q4": 4, "keep": True, "reason": "ok"})
    gemini_response = json.dumps({"q1": 2, "q2": 1, "q3": 2, "q4": 2, "keep": False, "reason": "bad"})

    mock_openai = MagicMock()
    mock_openai.generate.return_value = gpt_response
    mock_google = MagicMock()
    mock_google.generate.return_value = gemini_response

    qc_results_path = tmp_path / "qc_results.jsonl"
    flagged_path = tmp_path / "flagged.jsonl"

    results = run_phase5(
        combined_path=combined_path,
        openai_client=mock_openai,
        google_client=mock_google,
        qc_results_path=qc_results_path,
        flagged_path=flagged_path,
        seed=42,
    )

    assert results["flagged"] > 0
    assert flagged_path.exists()
    with open(flagged_path) as f:
        flagged_lines = f.readlines()
    assert len(flagged_lines) == results["flagged"]


def test_run_phase5_handles_unparseable_judge_response(tmp_path):
    """Phase 5 skips examples where a judge returns unparseable output."""
    from src.data.dpo.build_dpo_dataset import run_phase5

    combined_path = tmp_path / "dpo_combined.jsonl"
    ex = {
        "prompt": "<|L0_START|>rule<|L0_END|>\n<|L3_START|>msg<|L3_END|>",
        "chosen": "<|RESP_START|>" + "good " * 15 + "<|RESP_END|>",
        "rejected": "<|RESP_START|>" + "bad " * 15 + "<|RESP_END|>",
        "conflict_type": "L0_vs_L3",
        "category": "pairwise",
    }
    with open(combined_path, "w") as f:
        for i in range(10):
            f.write(json.dumps({**ex, "prompt": ex["prompt"] + f" {i}"}) + "\n")

    # GPT returns garbage, Gemini returns valid
    mock_openai = MagicMock()
    mock_openai.generate.return_value = "not json at all"
    mock_google = MagicMock()
    mock_google.generate.return_value = json.dumps(
        {"q1": 4, "q2": 4, "q3": 4, "q4": 4, "keep": True, "reason": "ok"}
    )

    qc_results_path = tmp_path / "qc_results.jsonl"
    flagged_path = tmp_path / "flagged.jsonl"

    results = run_phase5(
        combined_path=combined_path,
        openai_client=mock_openai,
        google_client=mock_google,
        qc_results_path=qc_results_path,
        flagged_path=flagged_path,
        seed=42,
    )

    # Unparseable examples should be counted as skipped
    assert "skipped" in results
