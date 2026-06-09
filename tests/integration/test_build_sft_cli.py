"""Integration tests for build_sft_dataset CLI."""

import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def run_cli(*args: str) -> subprocess.CompletedProcess:
    """Run build_sft_dataset.py with given arguments."""
    return subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "bin" / "build_sft_dataset.py"), *args],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )


def test_build_sft_cli_help():
    """--help exits 0 and mentions sft."""
    result = run_cli("--help")
    assert result.returncode == 0
    assert "sft" in result.stdout.lower()


def test_build_sft_cli_rejects_unknown_flags():
    """Unknown flags cause a non-zero exit code."""
    result = run_cli("--nonexistent")
    assert result.returncode != 0
