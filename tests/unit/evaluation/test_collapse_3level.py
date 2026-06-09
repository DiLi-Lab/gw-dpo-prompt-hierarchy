"""Tests for the --collapse-3level eval-input rewriter."""

import json
from pathlib import Path

import pytest

# Imported from the CLI module — re-exported via __all__ if needed.
from importlib import import_module

run_eval_mod = import_module("bin.run_evaluation")


def _make_5level_record(rec_id: str) -> dict:
    return {
        "id": rec_id,
        "prompt": (
            "<|L0_START|>rule<|L0_END|>\n"
            "<|L1_START|>persona<|L1_END|>\n"
            "<|L2_START|>config<|L2_END|>\n"
            "<|L3_START|>user-msg<|L3_END|>\n"
            "<|L4_START|>tool-out<|L4_END|>"
        ),
        "conflict_type": "L1_vs_L3",
        "victim_level": 1,
        "attacker_level": 3,
    }


def test_collapse_helper_rewrites_prompts(tmp_path):
    src = tmp_path / "eval_conflicts.jsonl"
    with src.open("w") as f:
        for i in range(3):
            f.write(json.dumps(_make_5level_record(f"id{i}")) + "\n")

    dst = run_eval_mod._maybe_collapse_inputs(
        output_dir=tmp_path,
        conflict_path=src,
        aligned_path=src,        # reuse — we only check structure
        reference_path=src,      # reuse — collapse is a no-op for non-delimited
        enable=True,
    )
    conflict_dst, aligned_dst, reference_dst = dst

    for path in (conflict_dst, aligned_dst):
        records = [json.loads(line) for line in path.open()]
        for r in records:
            assert "<|L1_START|>" not in r["prompt"]
            assert "<|L2_START|>" not in r["prompt"]
            assert "<|L0_START|>rule\n\npersona\n\nconfig<|L0_END|>" in r["prompt"]


def test_disabled_passthrough_returns_input_paths(tmp_path):
    src = tmp_path / "eval.jsonl"
    src.write_text(json.dumps(_make_5level_record("id0")) + "\n")
    out = run_eval_mod._maybe_collapse_inputs(
        output_dir=tmp_path,
        conflict_path=src,
        aligned_path=src,
        reference_path=src,
        enable=False,
    )
    assert out == (src, src, src)
