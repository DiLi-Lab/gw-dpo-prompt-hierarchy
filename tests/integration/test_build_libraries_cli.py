"""Integration tests for build_libraries CLI."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def run_cli(*args: str) -> subprocess.CompletedProcess:
    """Run build_libraries.py with given arguments."""
    return subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "bin" / "build_libraries.py"), *args],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )


def test_cli_no_subcommand():
    result = run_cli()
    assert result.returncode != 0


def test_cli_unknown_flag():
    result = run_cli("--unknown-flag")
    assert result.returncode != 0


def test_cli_l0_validate_missing_file(tmp_path):
    result = run_cli("l0", "--rules-file", str(tmp_path / "missing.json"))
    assert result.returncode != 0


def test_cli_l0_validate_valid_file(tmp_path):
    rules = [
        {"category": "system_integrity", "rule": "Test rule.", "id": "si_01"},
    ]
    rules_file = tmp_path / "l0_rules.json"
    rules_file.write_text(json.dumps(rules))
    result = run_cli("l0", "--validate", "--rules-file", str(rules_file))
    assert result.returncode == 0


def test_cli_l2_succeeds():
    result = run_cli("l2")
    assert result.returncode == 0


def test_cli_l2_validate():
    result = run_cli("l2", "--validate")
    assert result.returncode == 0
    assert "attribute" in result.stderr.lower()


def test_cli_l2_validate_with_count():
    result = run_cli("l2", "--validate", "--count", "50")
    assert result.returncode == 0
    assert "50" in result.stderr


def test_cli_injection_validate_missing_file(tmp_path):
    result = run_cli("injection", "--templates-file", str(tmp_path / "missing.json"))
    assert result.returncode != 0


def test_cli_l1_validate_valid_file(tmp_path):
    prompts = [
        {
            "persona": "Coder",
            "constraints": ["be concise"],
            "full_prompt": "You are a concise coder.",
            "domain": "coding",
            "batch_idx": 0,
        },
    ]
    lib_file = tmp_path / "l1_library.json"
    lib_file.write_text(json.dumps(prompts))
    result = run_cli("l1", "--validate", "--library-file", str(lib_file))
    assert result.returncode == 0
    assert "validated" in result.stderr.lower()


def test_cli_l1_validate_missing_file(tmp_path):
    result = run_cli("l1", "--validate", "--library-file", str(tmp_path / "missing.json"))
    assert result.returncode != 0
