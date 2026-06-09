"""Model architecture modifications for the 5-level instruction hierarchy.

Provides special token management, Instructional Segment Embeddings (ISE),
segment ID computation, and the LlamaWithISE model wrapper.
"""

from src.model.ise import InstructionalSegmentEmbedding
from src.model.llama_with_ise import LlamaWithISE, init_new_token_embeddings
from src.model.segment_ids import compute_segment_ids, compute_segment_ids_batch
from src.model.special_tokens import add_hierarchy_tokens

__all__ = [
    "InstructionalSegmentEmbedding",
    "LlamaWithISE",
    "add_hierarchy_tokens",
    "compute_segment_ids",
    "compute_segment_ids_batch",
    "init_new_token_embeddings",
]
