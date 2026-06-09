"""Llama model wrapper with Instructional Segment Embeddings.

Uses composition to wrap any LlamaForCausalLM with an ISE layer,
adding segment embeddings to token embeddings before the transformer.
"""

import logging

import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from src.config.constants import RESPONSE_SEGMENT_ID
from src.model.ise import InstructionalSegmentEmbedding

logger = logging.getLogger(__name__)


class LlamaWithISE(nn.Module):
    """Wraps a Llama model with optional Instructional Segment Embeddings.

    Args:
        model: A LlamaForCausalLM (or compatible) model.
        ise: An InstructionalSegmentEmbedding module, or ``None`` for the
            (f) tokens-only ablation. When ``None``, the wrapper is a
            transparent passthrough that ignores ``segment_ids`` and forwards
            ``input_ids`` directly to the inner model — no embedding lookup
            or addition is performed, so the forward output matches the bare
            inner model exactly.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        ise: InstructionalSegmentEmbedding | None,
    ) -> None:
        super().__init__()
        self.model = model
        self.ise = ise

    @property
    def config(self):
        return self.model.config

    @property
    def device(self):
        return self.model.device

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        segment_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        if self.ise is None:
            # Tokens-only ablation: pass through to the inner model with no
            # embedding lookup and no ISE addition. segment_ids is silently
            # discarded so collators/trainers shared with ISE-on configs do
            # not need a separate code path.
            return self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                inputs_embeds=inputs_embeds,
                **kwargs,
            )

        if inputs_embeds is None and input_ids is not None:
            inputs_embeds = self.model.get_input_embeddings()(input_ids)

        if segment_ids is not None and inputs_embeds is not None:
            seq_len = inputs_embeds.shape[1]
            seg_len = segment_ids.shape[1]
            if seg_len != seq_len:
                msg = (
                    "segment_ids length (%d) must match inputs_embeds "
                    "sequence length (%d). Ensure the data collator pads "
                    "segment_ids to the same length as input_ids."
                )
                raise ValueError(msg % (seg_len, seq_len))
            inputs_embeds = self.ise.add_to_embeddings(inputs_embeds, segment_ids)

        return self.model(
            input_ids=None,
            attention_mask=attention_mask,
            labels=labels,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

    def gradient_checkpointing_enable(self, **kwargs):
        self.model.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self):
        self.model.gradient_checkpointing_disable()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def get_output_embeddings(self):
        return self.model.get_output_embeddings()

    def add_model_tags(self, tags: list[str] | str) -> None:
        """Delegate to the wrapped model's ``PreTrainedModel.add_model_tags``.

        TRL's ``DPOTrainer.__init__`` calls ``self.model.add_model_tags(...)``
        to tag the model for Hub metadata. Without this forwarder the call
        raises ``AttributeError`` because ``LlamaWithISE`` is a plain
        ``nn.Module``. PEFT models expose ``add_model_tags`` via their own
        ``__getattr__`` forwarding to the base HF model.
        """
        self.model.add_model_tags(tags)


def init_new_token_embeddings(model: PreTrainedModel, num_new_tokens: int) -> None:
    """Initialize new token embeddings to the mean of existing embeddings.

    Applies to both the input embedding layer (embed_tokens) and the output
    projection (lm_head). StruQ found that mean initialisation preserves
    performance while random initialisation causes significant degradation.

    Args:
        model: Model with already-resized embedding matrix.
        num_new_tokens: Number of new tokens that were added at the end.
    """
    if num_new_tokens <= 0:
        return

    embed = model.get_input_embeddings()
    mean_emb = embed.weight.data[:-num_new_tokens].mean(dim=0)
    for i in range(1, num_new_tokens + 1):
        embed.weight.data[-i] = mean_emb.clone()

    lm_head = model.get_output_embeddings()
    if lm_head is not None:
        mean_lm = lm_head.weight.data[:-num_new_tokens].mean(dim=0)
        for i in range(1, num_new_tokens + 1):
            lm_head.weight.data[-i] = mean_lm.clone()

    logger.info(
        "Initialized %d new token embeddings to mean of existing embeddings",
        num_new_tokens,
    )
