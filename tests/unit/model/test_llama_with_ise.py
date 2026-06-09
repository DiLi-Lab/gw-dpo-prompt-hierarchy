"""Unit tests for the LlamaWithISE wrapper."""

import torch
from transformers import AutoModelForCausalLM

from src.model.ise import InstructionalSegmentEmbedding
from src.model.llama_with_ise import LlamaWithISE

MODEL_NAME = "hf-internal-testing/tiny-random-LlamaForCausalLM"


def _make_wrapper() -> LlamaWithISE:
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    ise = InstructionalSegmentEmbedding(
        num_segments=6, hidden_size=model.config.hidden_size,
    )
    return LlamaWithISE(model=model, ise=ise)


def _make_wrapper_no_ise() -> LlamaWithISE:
    """Wrapper with ise=None — the (f) tokens-only ablation configuration."""
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    return LlamaWithISE(model=model, ise=None)


def test_add_model_tags_delegates_to_inner_model():
    """TRL's DPOTrainer.__init__ calls model.add_model_tags(self._tag_names).

    The wrapper must forward this call to the inner PreTrainedModel so that
    tags land in its ``model_tags`` list (matching PreTrainedModel semantics).
    """
    wrapped = _make_wrapper()

    wrapped.add_model_tags(["gw-dpo", "ise"])

    assert wrapped.model.model_tags == ["gw-dpo", "ise"]


def test_add_model_tags_accepts_string():
    wrapped = _make_wrapper()

    wrapped.add_model_tags("gw-dpo")

    assert wrapped.model.model_tags == ["gw-dpo"]


def test_add_model_tags_does_not_duplicate():
    wrapped = _make_wrapper()

    wrapped.add_model_tags(["gw-dpo"])
    wrapped.add_model_tags(["gw-dpo", "ise"])

    assert wrapped.model.model_tags == ["gw-dpo", "ise"]


def test_no_ise_forward_matches_inner_model():
    """With ise=None, the wrapper must be a transparent passthrough.

    The (f) tokens-only ablation runs SFT/DPO without ISE while still
    using the wrapper for code uniformity. Forward outputs must match the
    inner model exactly so ablation results attribute solely to the
    presence/absence of segment embeddings, not to a stale wrapper code path.
    """
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    wrapped = LlamaWithISE(model=model, ise=None)

    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    attention_mask = torch.ones_like(input_ids)
    segment_ids = torch.zeros_like(input_ids)  # should be ignored when ise=None

    with torch.no_grad():
        wrapped_out = wrapped(
            input_ids=input_ids,
            attention_mask=attention_mask,
            segment_ids=segment_ids,
        )
        bare_out = model(input_ids=input_ids, attention_mask=attention_mask)

    assert torch.allclose(wrapped_out.logits, bare_out.logits)


def test_no_ise_forward_ignores_segment_ids():
    """segment_ids passed when ise=None must not raise nor affect output."""
    torch.manual_seed(0)
    wrapped = _make_wrapper_no_ise()
    input_ids = torch.tensor([[1, 2, 3]])

    with torch.no_grad():
        out_with_segs = wrapped(
            input_ids=input_ids,
            segment_ids=torch.tensor([[2, 3, 4]]),
        )
        out_without_segs = wrapped(input_ids=input_ids)

    assert torch.allclose(out_with_segs.logits, out_without_segs.logits)
