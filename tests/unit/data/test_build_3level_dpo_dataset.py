"""Smoke test for bin/build_3level_dpo_dataset.py.

Builds a synthetic 5-record fixture covering each pair-class
(intra-System -> dropped, calibration -> kept, gap-1 cross, gap-2 cross,
no-L4) and asserts that the build script's outputs match expectations.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]


def _record(victim: int, attacker: int, *, has_l4: bool = True,
            is_calibration: bool = False, conflict_type: str = "synthetic",
            level_gap: int | None = None) -> dict:
    parts = [
        "<|L0_START|>rule<|L0_END|>",
        "<|L1_START|>persona<|L1_END|>",
        "<|L2_START|>config<|L2_END|>",
        "<|L3_START|>user-msg<|L3_END|>",
    ]
    if has_l4:
        parts.append("<|L4_START|>tool-out<|L4_END|>")
    return {
        "prompt": "\n".join(parts),
        "chosen": "<|RESP_START|>good<|RESP_END|>",
        "rejected": "<|RESP_START|>bad<|RESP_END|>",
        "conflict_type": conflict_type,
        "level_gap": level_gap if level_gap is not None else abs(attacker - victim),
        "margin": 0.0,
        "category": "calibration" if is_calibration else "pairwise",
        "is_calibration": is_calibration,
        "victim_level": victim,
        "attacker_level": attacker,
        "levels_present": [0, 1, 2, 3] + ([4] if has_l4 else []),
    }


@pytest.fixture
def fixture_paths(tmp_path):
    train_dir = tmp_path / "data" / "dpo" / "train"
    val_dir = tmp_path / "data" / "dpo" / "val"
    train_dir.mkdir(parents=True)
    val_dir.mkdir(parents=True)
    rows = [
        _record(0, 1, conflict_type="L0_vs_L1"),               # dropped (intra-System)
        _record(0, 4, conflict_type="L0_vs_L4"),               # kept, gap=2
        _record(1, 3, conflict_type="L1_vs_L3", has_l4=False), # kept, gap=1, no L4
        _record(3, 4, conflict_type="L3_vs_L4"),               # kept, gap=1
        _record(3, 3, conflict_type="calibration_L3",          # kept, gap=0
                is_calibration=True, level_gap=0),
    ]
    for path in (train_dir / "dpo_combined.jsonl",
                 val_dir / "dpo_combined.jsonl"):
        with path.open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    return tmp_path, rows


def test_build_drops_intra_system_and_collapses_prompts(fixture_paths):
    tmp_path, rows = fixture_paths
    cmd = [
        sys.executable,
        str(REPO_ROOT / "bin" / "build_3level_dpo_dataset.py"),
        "--project-root", str(tmp_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    assert proc.returncode == 0, proc.stderr

    for split in ("train", "val"):
        out = tmp_path / "data" / "dpo" / f"{split}_3level" / "dpo_combined.jsonl"
        assert out.exists(), f"missing {out}"
        records = [json.loads(line) for line in out.open()]

        # 4 of 5 kept (intra-System dropped).
        assert len(records) == 4

        # Original conflict types preserved as provenance.
        kept_types = {r["original_conflict_type"] for r in records}
        assert "L0_vs_L1" not in kept_types

        for r in records:
            # Prompts collapsed: no L1/L2 wrappers.
            assert "<|L1_START|>" not in r["prompt"]
            assert "<|L2_START|>" not in r["prompt"]
            # Provenance fields present.
            assert "source_5level_id" in r
            assert "collapse_version" in r
            assert "original_victim_level" in r
            assert "original_attacker_level" in r
            # Level gaps recomputed under 3-level.
            v, a = r["original_victim_level"], r["original_attacker_level"]
            if r["is_calibration"]:
                assert r["level_gap"] == 0
            elif (v, a) in {(0, 4), (4, 0)}:
                assert r["level_gap"] == 2
            else:
                assert r["level_gap"] == 1
            # Margin matches level_gap (trainer multiplies by alpha).
            assert r["margin"] == float(r["level_gap"])
