"""End-to-end smoke test for the DPO HP-search driver."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT = PROJECT_ROOT / "bin" / "train_dpo_hp_search.py"

_requires_artifacts = pytest.mark.skipif(
    not Path(PROJECT_ROOT / "test-run" / "models" / "llama-3.2-1b-sft-merged"
              / "config.json").exists(),
    reason="Requires test-run SFT artifacts (run bin/run_test.sh --setup-only first)",
)


def run_driver(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT),
          "--config", str(PROJECT_ROOT / "configs" / "test.yaml"), *args],
        capture_output=True, text=True, cwd=str(PROJECT_ROOT), timeout=1800,
    )


def test_cli_rejects_unknown_flag():
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--nonexistent"],
        capture_output=True, text=True, cwd=str(PROJECT_ROOT), timeout=10,
    )
    assert result.returncode != 0
    assert "unrecognized" in result.stderr.lower()


@_requires_artifacts
def test_smoke_single_config(tmp_path, monkeypatch):
    result = run_driver("--configs", "1", "--override", "dpo.num_curriculum_stages=1")
    assert result.returncode == 0, result.stderr[-2000:]

    hp_root = PROJECT_ROOT / "test-run" / "models" / "hp_search"
    assert (hp_root / "data" / "hp_select.jsonl").exists()
    assert (hp_root / "data" / "val_train.jsonl").exists()
    assert (hp_root / "results.jsonl").exists()
    assert (hp_root / "best_config.json").exists()

    results = [json.loads(l) for l in open(hp_root / "results.jsonl")]
    assert len(results) >= 1
    assert {"config_id", "rho", "beta", "alpha", "hp_select"} <= set(results[0])


@_requires_artifacts
def test_resumes_when_hp_eval_exists():
    hp_root = PROJECT_ROOT / "test-run" / "models" / "hp_search"
    if not (hp_root / "results.jsonl").exists():
        pytest.skip("Requires prior successful run")

    result = run_driver("--configs", "1", "--override", "dpo.num_curriculum_stages=1")
    assert result.returncode == 0
    assert ("skipping" in result.stdout.lower()
             or "skipping" in result.stderr.lower()), result.stderr[-500:]
