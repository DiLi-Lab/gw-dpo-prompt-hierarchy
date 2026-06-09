"""Integration tests for DPO dataset CLI."""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "bin" / "build_dpo_dataset.py"), *args],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
        timeout=60,
    )


def test_cli_rejects_unknown_flags():
    result = run_cli("--nonexistent-flag")
    assert result.returncode != 0
    assert "unrecognized" in result.stderr.lower()


def test_cli_dry_run():
    result = run_cli("--dry-run")
    assert result.returncode == 0
    assert "dry run" in result.stderr.lower() or "dry run" in result.stdout.lower()


def test_cli_help():
    result = run_cli("--help")
    assert result.returncode == 0
    assert "phase" in result.stdout.lower()
