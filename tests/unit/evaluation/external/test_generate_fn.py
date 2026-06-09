"""Smoke-test the lifted ISE-aware generate-batch closure.

Uses lightweight stand-ins (no real model load) to verify that the
function signature and dispatch logic are preserved during the lift
from bin/run_evaluation.py.
"""

from unittest.mock import MagicMock

import torch

from src.evaluation.external.generate import build_generate_fn


def _fake_tokenizer() -> MagicMock:
    tok = MagicMock()
    tok.pad_token_id = 0
    tok.eos_token_id = 1
    tok.unk_token_id = 100
    tok.convert_tokens_to_ids.return_value = 999  # RESP_END id
    tok.return_value = {
        "input_ids": torch.tensor([[5, 6, 7], [8, 9, 10]]),
        "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 1]]),
    }
    tok.decode.side_effect = lambda ids, skip_special_tokens: f"<decoded-{len(ids)}>"
    return tok


def test_no_ise_path_returns_one_response_per_prompt() -> None:
    tok = _fake_tokenizer()
    model = MagicMock()
    model.parameters.return_value = iter([torch.tensor([1.0])])
    # Output: pretend generate returned input + 4 new tokens for each row.
    model.generate.return_value = torch.tensor([
        [5, 6, 7, 11, 12, 13, 14],
        [8, 9, 10, 15, 16, 17, 18],
    ])

    gen_fn = build_generate_fn(
        model, tok, has_ise=False,
        max_new_tokens=8, temperature=0.0,
    )
    out = gen_fn(["prompt-a", "prompt-b"])
    assert len(out) == 2
    model.generate.assert_called_once()
