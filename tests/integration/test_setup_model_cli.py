"""Integration tests for bin/setup_model.py CLI script."""

import subprocess
import sys
from pathlib import Path

import pytest
import torch
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _hf_auth_available() -> bool:
    """Check whether HuggingFace authentication is configured."""
    try:
        from huggingface_hub import HfApi
        HfApi().whoami()
        return True
    except Exception:
        return False


@pytest.mark.skipif(
    not _hf_auth_available(),
    reason="HuggingFace authentication not available (required for gated Llama model)",
)
def test_setup_model_creates_artifacts(tmp_path: Path) -> None:
    """Run setup_model.py with tiny model and verify output files."""
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "bin" / "setup_model.py"),
            "--config",
            str(PROJECT_ROOT / "configs" / "base_linear.yaml"),
            "--override",
            f"paths.project_root={tmp_path}",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"setup_model.py failed:\nstdout={result.stdout}\nstderr={result.stderr}"
    )

    # Verify tokenizer directory
    tokenizer_dir = tmp_path / "models" / "tokenizer-5level"
    assert tokenizer_dir.exists()
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))
    assert "<|L0_START|>" in tokenizer.get_vocab()
    assert "<|RESP_END|>" in tokenizer.get_vocab()

    # Verify ISE weights
    ise_path = tmp_path / "models" / "ise_weights_init.pt"
    assert ise_path.exists()
    ise_state = torch.load(ise_path, weights_only=True)
    assert "segment_embedding.weight" in ise_state
    # Normal-initialized (near zero, std=0.01)
    assert ise_state["segment_embedding.weight"].abs().max() < 0.5

    # Verify model directory
    model_dir = tmp_path / "models" / "base-with-tokens"
    assert model_dir.exists()
    assert (model_dir / "config.json").exists()
    # Tokenizer also saved alongside model
    model_tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    assert "<|L0_START|>" in model_tokenizer.get_vocab()


def test_setup_model_rejects_unknown_flags() -> None:
    """Argparse should reject unknown flags."""
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "bin" / "setup_model.py"),
            "--nonexistent-flag",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0
    assert "unrecognized arguments" in result.stderr
