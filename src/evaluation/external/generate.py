"""ISE-aware generate-batch closure shared across eval CLIs.

Lifted verbatim from ``bin/run_evaluation.py`` so the 5-level suite and
the new external-benchmark CLIs share one implementation.
"""

from typing import Callable

import torch
from transformers import PreTrainedTokenizerBase

from src.model.llama_with_ise import LlamaWithISE
from src.model.segment_ids import compute_segment_ids_batch


def build_generate_fn(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    has_ise: bool,
    max_new_tokens: int,
    temperature: float,
) -> Callable[[list[str]], list[str]]:
    """Return a callable taking list[str] -> list[str]."""
    device = next(model.parameters()).device
    do_sample = temperature > 0.0

    # The merged DPO model carries a generation_config with do_sample=True,
    # temperature=0.6, top_p=0.9 (Llama-3.1 chat-template defaults baked in
    # by training). When we override do_sample=False at eval time,
    # transformers 5 emits a "generation flags not valid and may be ignored"
    # warning per call. Reset the inherited sampling fields to do_sample=False
    # canonical values once at startup so the warning never fires.
    inner_model = model.model if isinstance(model, LlamaWithISE) else model
    if not do_sample and hasattr(inner_model, "generation_config"):
        gc = inner_model.generation_config
        gc.do_sample = False
        gc.temperature = 1.0
        gc.top_p = 1.0
        # transformers 5 flags top_k=0 as "not valid" for greedy decoding;
        # None drops the field entirely from the validation pass.
        gc.top_k = None

    # We do NOT add <|RESP_END|> to eos_token_id. An earlier version of this
    # code did, but it caused the SFT and (untrained) baseline models to
    # produce ~50% empty responses on the conflict split: when the model
    # emits RESP_END as one of its first generated tokens (which happens
    # frequently on hard / unfamiliar-format prompts), generation stops
    # before any content is produced and the truncated output decodes to "".
    # Standard Llama EOS tokens are sufficient for stopping; the
    # post-decode truncation below cleans up any RESP_END drift.
    resp_end_id = tokenizer.convert_tokens_to_ids("<|RESP_END|>")

    def _truncate_at_resp_end(seq):
        """Drop everything from <|RESP_END|> onward.

        The trained models learn to emit RESP_END at the end of a response
        but it isn't in the model's eos_token_id list, so generate()
        continues past it. Without this truncation, drift past RESP_END
        (a few extra tokens to the next standard Llama EOS) would be
        glommed into the response string by skip_special_tokens=True
        decoding and shown to the judge.
        """
        if resp_end_id is None or resp_end_id == tokenizer.unk_token_id:
            return seq
        # seq: 1D tensor of token IDs.
        matches = (seq == resp_end_id).nonzero(as_tuple=True)[0]
        if len(matches) > 0:
            return seq[: int(matches[0])]
        return seq

    def gen_batch(prompts: list[str]) -> list[str]:
        encoded = tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=False,
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        gen_kwargs: dict = {
            "attention_mask": attention_mask,
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.pad_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature

        target = model.model if isinstance(model, LlamaWithISE) else model

        if has_ise:
            seg = compute_segment_ids_batch(
                [input_ids[i].tolist() for i in range(input_ids.shape[0])],
                tokenizer,
            ).to(device)
            seg_emb = model.ise(seg)
            tok_emb = model.model.get_input_embeddings()(input_ids)
            inputs_embeds = tok_emb + seg_emb
            gen_kwargs["inputs_embeds"] = inputs_embeds
            out = target.generate(**gen_kwargs)
            # When using inputs_embeds, generate() returns only new tokens.
            responses: list[str] = []
            for i in range(out.shape[0]):
                trimmed = _truncate_at_resp_end(out[i])
                responses.append(
                    tokenizer.decode(trimmed, skip_special_tokens=True),
                )
            return responses

        gen_kwargs["input_ids"] = input_ids
        out = target.generate(**gen_kwargs)
        responses = []
        for i in range(out.shape[0]):
            new_tokens = out[i, input_ids.shape[1]:]
            new_tokens = _truncate_at_resp_end(new_tokens)
            responses.append(
                tokenizer.decode(new_tokens, skip_special_tokens=True),
            )
        return responses

    return gen_batch
