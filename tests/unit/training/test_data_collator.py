"""Tests for the hierarchy-aware data collator."""

import torch
from transformers import AutoTokenizer

from src.model.special_tokens import add_hierarchy_tokens
from src.training.data_collator import HierarchyDataCollator


def _make_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("models/tokenizer-5level")
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _make_example(tokenizer, text: str) -> dict:
    encoding = tokenizer(text, truncation=True, max_length=4096)
    return {
        "input_ids": encoding["input_ids"],
        "attention_mask": encoding["attention_mask"],
    }


def test_collator_returns_expected_keys():
    tokenizer = _make_tokenizer()
    collator = HierarchyDataCollator(tokenizer=tokenizer, max_seq_length=4096)
    text = "<|L0_START|>Be helpful.<|L0_END|><|RESP_START|>OK<|RESP_END|>"
    example = _make_example(tokenizer, text)
    batch = collator([example])
    assert "input_ids" in batch
    assert "attention_mask" in batch
    assert "labels" in batch
    assert "segment_ids" in batch


def test_collator_output_shapes_match():
    tokenizer = _make_tokenizer()
    collator = HierarchyDataCollator(tokenizer=tokenizer, max_seq_length=4096)
    text = "<|L0_START|>Be helpful.<|L0_END|><|RESP_START|>OK<|RESP_END|>"
    example = _make_example(tokenizer, text)
    batch = collator([example, example])
    assert batch["input_ids"].shape == batch["segment_ids"].shape
    assert batch["input_ids"].shape == batch["attention_mask"].shape
    assert batch["input_ids"].shape == batch["labels"].shape


def test_collator_masks_prompt_tokens():
    tokenizer = _make_tokenizer()
    collator = HierarchyDataCollator(tokenizer=tokenizer, max_seq_length=4096)
    text = "<|L0_START|>Be helpful.<|L0_END|><|RESP_START|>OK<|RESP_END|>"
    example = _make_example(tokenizer, text)
    batch = collator([example])

    resp_start_id = tokenizer.convert_tokens_to_ids("<|RESP_START|>")
    input_ids = batch["input_ids"][0].tolist()
    labels = batch["labels"][0].tolist()

    resp_pos = input_ids.index(resp_start_id)
    for i in range(resp_pos + 1):
        assert labels[i] == -100, f"Label at position {i} should be -100, got {labels[i]}"
    response_labels = [l for l in labels[resp_pos + 1:] if l != -100]
    assert len(response_labels) > 0, "No response tokens in labels"


def test_collator_segment_ids_correct():
    tokenizer = _make_tokenizer()
    collator = HierarchyDataCollator(tokenizer=tokenizer, max_seq_length=4096)
    text = "<|L0_START|>Rule.<|L0_END|><|L1_START|>System.<|L1_END|><|RESP_START|>Response.<|RESP_END|>"
    example = _make_example(tokenizer, text)
    batch = collator([example])
    segment_ids = batch["segment_ids"][0].tolist()
    assert 0 in segment_ids
    assert 1 in segment_ids
    assert 5 in segment_ids


def test_collator_pads_to_batch_max():
    tokenizer = _make_tokenizer()
    collator = HierarchyDataCollator(tokenizer=tokenizer, max_seq_length=4096)
    short = "<|L0_START|>A.<|L0_END|><|RESP_START|>B<|RESP_END|>"
    long = "<|L0_START|>A very long rule about many things.<|L0_END|><|L1_START|>System prompt here.<|L1_END|><|RESP_START|>A longer response.<|RESP_END|>"
    batch = collator([_make_example(tokenizer, short), _make_example(tokenizer, long)])
    assert batch["input_ids"].shape[0] == 2
    assert batch["input_ids"].shape[1] == batch["segment_ids"].shape[1]


def test_collator_truncates_to_max_seq_length():
    tokenizer = _make_tokenizer()
    collator = HierarchyDataCollator(tokenizer=tokenizer, max_seq_length=32)
    text = "<|L0_START|>" + "word " * 100 + "<|L0_END|><|RESP_START|>OK<|RESP_END|>"
    example = _make_example(tokenizer, text)
    batch = collator([example])
    assert batch["input_ids"].shape[1] <= 32
