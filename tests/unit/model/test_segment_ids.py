"""Tests for segment ID computation from delimiter tokens."""

import torch
from transformers import AutoTokenizer

from src.config.constants import RESPONSE_SEGMENT_ID
from src.model.segment_ids import compute_segment_ids, compute_segment_ids_batch
from src.model.special_tokens import add_hierarchy_tokens


def _make_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(
        "hf-internal-testing/tiny-random-LlamaForCausalLM"
    )
    add_hierarchy_tokens(tokenizer)
    return tokenizer


def test_full_five_level_prompt():
    tokenizer = _make_tokenizer()
    text = (
        "<|L0_START|>rule<|L0_END|>"
        "<|L1_START|>system<|L1_END|>"
        "<|L2_START|>config<|L2_END|>"
        "<|L3_START|>query<|L3_END|>"
        "<|L4_START|>data<|L4_END|>"
        "<|RESP_START|>answer<|RESP_END|>"
    )
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    segment_ids = compute_segment_ids(token_ids, tokenizer)

    assert len(segment_ids) == len(token_ids)
    for seg_id in segment_ids:
        assert 0 <= seg_id <= RESPONSE_SEGMENT_ID


def test_segment_ids_for_delimiters():
    tokenizer = _make_tokenizer()
    text = "<|L0_START|>hello<|L0_END|><|L1_START|>world<|L1_END|>"
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    segment_ids = compute_segment_ids(token_ids, tokenizer)

    l0_start_id = tokenizer.convert_tokens_to_ids("<|L0_START|>")
    l0_end_id = tokenizer.convert_tokens_to_ids("<|L0_END|>")
    l1_start_id = tokenizer.convert_tokens_to_ids("<|L1_START|>")
    l1_end_id = tokenizer.convert_tokens_to_ids("<|L1_END|>")

    for tid, seg in zip(token_ids, segment_ids):
        if tid == l0_start_id or tid == l0_end_id:
            assert seg == 0
        elif tid == l1_start_id or tid == l1_end_id:
            assert seg == 1


def test_response_segment():
    tokenizer = _make_tokenizer()
    text = "<|RESP_START|>response text<|RESP_END|>"
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    segment_ids = compute_segment_ids(token_ids, tokenizer)

    for seg in segment_ids:
        assert seg == RESPONSE_SEGMENT_ID


def test_default_segment_is_response():
    tokenizer = _make_tokenizer()
    text = "some text with no delimiters"
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    segment_ids = compute_segment_ids(token_ids, tokenizer)

    for seg in segment_ids:
        assert seg == RESPONSE_SEGMENT_ID


def test_partial_levels():
    tokenizer = _make_tokenizer()
    text = "<|L1_START|>system<|L1_END|><|L3_START|>query<|L3_END|>"
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    segment_ids = compute_segment_ids(token_ids, tokenizer)

    l1_start_id = tokenizer.convert_tokens_to_ids("<|L1_START|>")
    l1_end_id = tokenizer.convert_tokens_to_ids("<|L1_END|>")
    l3_start_id = tokenizer.convert_tokens_to_ids("<|L3_START|>")
    l3_end_id = tokenizer.convert_tokens_to_ids("<|L3_END|>")

    for tid, seg in zip(token_ids, segment_ids):
        if tid == l1_start_id or tid == l1_end_id:
            assert seg == 1
        elif tid == l3_start_id or tid == l3_end_id:
            assert seg == 3


def test_batch_computation():
    tokenizer = _make_tokenizer()
    texts = [
        "<|L0_START|>rule<|L0_END|><|RESP_START|>ok<|RESP_END|>",
        "<|L1_START|>sys<|L1_END|><|RESP_START|>ok<|RESP_END|>",
    ]
    token_id_lists = [
        tokenizer.encode(t, add_special_tokens=False) for t in texts
    ]

    expected = [compute_segment_ids(tids, tokenizer) for tids in token_id_lists]
    batch_result = compute_segment_ids_batch(token_id_lists, tokenizer)
    assert isinstance(batch_result, torch.Tensor)
    assert batch_result.shape[0] == 2

    for i, (exp, tids) in enumerate(zip(expected, token_id_lists)):
        for j in range(len(tids)):
            assert batch_result[i, j].item() == exp[j]
