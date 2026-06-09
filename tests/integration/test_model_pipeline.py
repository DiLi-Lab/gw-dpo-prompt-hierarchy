"""Integration test: full model pipeline with special tokens + ISE."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.model.ise import InstructionalSegmentEmbedding
from src.model.llama_with_ise import LlamaWithISE, init_new_token_embeddings
from src.model.segment_ids import compute_segment_ids_batch
from src.model.special_tokens import add_hierarchy_tokens

MODEL_NAME = "hf-internal-testing/tiny-random-LlamaForCausalLM"


def test_full_pipeline_forward_pass():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    tokenizer, num_added = add_hierarchy_tokens(tokenizer)
    assert num_added == 12

    model.resize_token_embeddings(len(tokenizer))
    init_new_token_embeddings(model, num_added)

    hidden_size = model.config.hidden_size
    ise = InstructionalSegmentEmbedding(num_segments=6, hidden_size=hidden_size)
    wrapped = LlamaWithISE(model=model, ise=ise)

    text = (
        "<|L0_START|>Be helpful<|L0_END|>"
        "<|L1_START|>You are an assistant<|L1_END|>"
        "<|RESP_START|>Hello<|RESP_END|>"
    )
    token_ids = tokenizer.encode(text)
    input_ids = torch.tensor([token_ids])
    segment_ids = compute_segment_ids_batch([token_ids], tokenizer)

    with torch.no_grad():
        outputs = wrapped(input_ids=input_ids, segment_ids=segment_ids)

    assert outputs.logits is not None
    assert outputs.logits.shape[0] == 1
    assert outputs.logits.shape[1] == input_ids.shape[1]
    assert outputs.logits.shape[2] == len(tokenizer)


def test_forward_without_segment_ids():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    tokenizer, num_added = add_hierarchy_tokens(tokenizer)
    model.resize_token_embeddings(len(tokenizer))
    init_new_token_embeddings(model, num_added)

    hidden_size = model.config.hidden_size
    ise = InstructionalSegmentEmbedding(num_segments=6, hidden_size=hidden_size)
    wrapped = LlamaWithISE(model=model, ise=ise)

    input_ids = tokenizer.encode("Hello world", return_tensors="pt")
    with torch.no_grad():
        outputs = wrapped(input_ids=input_ids)

    assert outputs.logits is not None


def test_mean_init_embeddings():
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    original_embed = model.get_input_embeddings()
    original_mean = original_embed.weight.data.mean(dim=0).clone()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer, num_added = add_hierarchy_tokens(tokenizer)
    model.resize_token_embeddings(len(tokenizer))
    init_new_token_embeddings(model, num_added)

    new_embed = model.get_input_embeddings()
    for i in range(1, num_added + 1):
        assert torch.allclose(
            new_embed.weight.data[-i], original_mean, atol=1e-5
        )

    lm_head = model.get_output_embeddings()
    original_lm_mean = lm_head.weight.data[:-num_added].mean(dim=0)
    for i in range(1, num_added + 1):
        assert torch.allclose(
            lm_head.weight.data[-i], original_lm_mean, atol=1e-5
        )


def test_save_and_load_ise(tmp_path):
    hidden_size = 32
    ise = InstructionalSegmentEmbedding(num_segments=6, hidden_size=hidden_size)
    with torch.no_grad():
        ise.segment_embedding.weight.fill_(1.0)

    ise_path = tmp_path / "ise_weights.pt"
    torch.save(ise.state_dict(), ise_path)

    ise2 = InstructionalSegmentEmbedding(num_segments=6, hidden_size=hidden_size)
    ise2.load_state_dict(torch.load(ise_path, weights_only=True))

    assert torch.allclose(ise.segment_embedding.weight, ise2.segment_embedding.weight)
