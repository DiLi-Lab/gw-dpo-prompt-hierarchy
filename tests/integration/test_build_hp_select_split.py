"""Integration tests for the HP-select split CLI."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT = PROJECT_ROOT / "bin" / "build_hp_select_split.py"


def _write_tiny_val(path: Path) -> None:
    records = []
    for i in range(20):
        records.append({"prompt": f"p{i}", "chosen": "c", "rejected": "r",
                         "level_gap": 0, "is_calibration": True, "margin": 0.0})
    for gap, n in [(1, 10), (2, 10), (3, 10), (4, 10)]:
        for i in range(n):
            records.append({"prompt": f"g{gap}_{i}", "chosen": "c", "rejected": "r",
                             "level_gap": gap, "is_calibration": False,
                             "margin": float(gap)})
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def run_cli(*args: str, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True, text=True, cwd=str(cwd), timeout=60,
    )


def test_cli_rejects_unknown_flag(tmp_path):
    result = run_cli("--nonexistent", cwd=tmp_path)
    assert result.returncode != 0
    assert "unrecognized" in result.stderr.lower()


def test_cli_builds_split(tmp_path):
    val = tmp_path / "data" / "dpo" / "val" / "dpo_combined.jsonl"
    _write_tiny_val(val)
    out_dir = tmp_path / "models" / "hp_search" / "data"
    result = run_cli("--source", str(val), "--out-dir", str(out_dir),
                      "--target-size", "20", "--seed", "42", cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert (out_dir / "hp_select.jsonl").exists()
    assert (out_dir / "val_train.jsonl").exists()
    assert (out_dir / "split_manifest.json").exists()

    hp_lines = (out_dir / "hp_select.jsonl").read_text().splitlines()
    val_lines = (out_dir / "val_train.jsonl").read_text().splitlines()
    assert len(hp_lines) == 20
    assert len(hp_lines) + len(val_lines) == 60

    manifest = json.loads((out_dir / "split_manifest.json").read_text())
    assert manifest["seed"] == 42
    assert manifest["hp_select_size"] == 20
    assert manifest["val_train_size"] == 40
    assert "source_sha256" in manifest


def test_cli_idempotent_when_source_unchanged(tmp_path):
    val = tmp_path / "data" / "dpo" / "val" / "dpo_combined.jsonl"
    _write_tiny_val(val)
    out_dir = tmp_path / "models" / "hp_search" / "data"

    r1 = run_cli("--source", str(val), "--out-dir", str(out_dir),
                  "--target-size", "20", "--seed", "42", cwd=tmp_path)
    assert r1.returncode == 0
    mtime1 = (out_dir / "hp_select.jsonl").stat().st_mtime

    r2 = run_cli("--source", str(val), "--out-dir", str(out_dir),
                  "--target-size", "20", "--seed", "42", cwd=tmp_path)
    assert r2.returncode == 0
    assert "already built" in (r2.stdout + r2.stderr).lower()
    mtime2 = (out_dir / "hp_select.jsonl").stat().st_mtime
    assert mtime1 == mtime2


def test_cli_rebuilds_when_source_changes(tmp_path):
    val = tmp_path / "data" / "dpo" / "val" / "dpo_combined.jsonl"
    _write_tiny_val(val)
    out_dir = tmp_path / "models" / "hp_search" / "data"

    run_cli("--source", str(val), "--out-dir", str(out_dir),
             "--target-size", "20", "--seed", "42", cwd=tmp_path)
    original_hp = (out_dir / "hp_select.jsonl").read_text()

    with open(val, "a") as f:
        f.write(json.dumps({"prompt": "new", "chosen": "c", "rejected": "r",
                              "level_gap": 0, "is_calibration": True,
                              "margin": 0.0}) + "\n")

    run_cli("--source", str(val), "--out-dir", str(out_dir),
             "--target-size", "20", "--seed", "42", cwd=tmp_path)
    new_hp = (out_dir / "hp_select.jsonl").read_text()
    assert original_hp != new_hp or (out_dir / "split_manifest.json").stat().st_size > 0
