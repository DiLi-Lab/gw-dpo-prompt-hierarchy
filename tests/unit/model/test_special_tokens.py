"""Tests for special token addition to tokenizers."""

from transformers import AutoTokenizer

from src.config.constants import SPECIAL_TOKENS
from src.model.special_tokens import add_hierarchy_tokens


def test_adds_correct_number_of_tokens():
    tokenizer = AutoTokenizer.from_pretrained(
        "hf-internal-testing/tiny-random-LlamaForCausalLM"
    )
    original_size = len(tokenizer)
    updated_tokenizer, num_added = add_hierarchy_tokens(tokenizer)
    assert num_added == 12
    assert len(updated_tokenizer) == original_size + 12


def test_tokens_are_single_ids():
    tokenizer = AutoTokenizer.from_pretrained(
        "hf-internal-testing/tiny-random-LlamaForCausalLM"
    )
    updated_tokenizer, _ = add_hierarchy_tokens(tokenizer)
    for token in SPECIAL_TOKENS:
        ids = updated_tokenizer.encode(token, add_special_tokens=False)
        assert len(ids) == 1, f"{token} should encode to a single ID, got {ids}"


def test_round_trip_encode_decode():
    tokenizer = AutoTokenizer.from_pretrained(
        "hf-internal-testing/tiny-random-LlamaForCausalLM"
    )
    updated_tokenizer, _ = add_hierarchy_tokens(tokenizer)
    for token in SPECIAL_TOKENS:
        token_id = updated_tokenizer.convert_tokens_to_ids(token)
        decoded = updated_tokenizer.convert_ids_to_tokens(token_id)
        assert decoded == token


def test_idempotent():
    tokenizer = AutoTokenizer.from_pretrained(
        "hf-internal-testing/tiny-random-LlamaForCausalLM"
    )
    updated, num1 = add_hierarchy_tokens(tokenizer)
    updated2, num2 = add_hierarchy_tokens(updated)
    assert num1 == 12
    assert num2 == 0


def test_save_and_reload(tmp_path):
    tokenizer = AutoTokenizer.from_pretrained(
        "hf-internal-testing/tiny-random-LlamaForCausalLM"
    )
    updated, _ = add_hierarchy_tokens(tokenizer)
    updated.save_pretrained(str(tmp_path / "tokenizer"))

    reloaded = AutoTokenizer.from_pretrained(str(tmp_path / "tokenizer"))
    for token in SPECIAL_TOKENS:
        assert reloaded.convert_tokens_to_ids(token) != reloaded.unk_token_id
