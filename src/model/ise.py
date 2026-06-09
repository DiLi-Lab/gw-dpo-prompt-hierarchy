"""Instructional Segment Embeddings (ISE) module.

Adds a learnable segment embedding to each token based on its hierarchy level.
Formally: final_embedding[m] = E_Tok[x_m] + E_Seg[h_m]
where h_m in {0,1,2,3,4,5} is the segment ID for token position m.

Initialized from a normal distribution (mean=0, std=0.01) so the model starts
close to the pretrained checkpoint while providing gradient signal.
Adds only num_segments * hidden_size parameters
(e.g., 6 * 4096 = 24,576 for Llama-3.1-8B).
"""

import torch
import torch.nn as nn


class InstructionalSegmentEmbedding(nn.Module):
    """Learnable segment embeddings for hierarchy-level tagging.

    Args:
        num_segments: Number of distinct segments (6: L0-L4 + response).
        hidden_size: Embedding dimension matching the model's hidden size.
        init_std: Standard deviation for normal initialization (default 0.01).
    """

    def __init__(
        self,
        num_segments: int,
        hidden_size: int,
        init_std: float = 0.01,
    ) -> None:
        super().__init__()
        self.num_segments = num_segments
        self.hidden_size = hidden_size
        self.segment_embedding = nn.Embedding(num_segments, hidden_size)
        nn.init.normal_(self.segment_embedding.weight, mean=0.0, std=init_std)

    def forward(self, segment_ids: torch.Tensor) -> torch.Tensor:
        """Look up segment embeddings.

        Args:
            segment_ids: Integer tensor of shape (batch_size, seq_len).

        Returns:
            Segment embeddings of shape (batch_size, seq_len, hidden_size).
        """
        return self.segment_embedding(segment_ids)

    def add_to_embeddings(
        self,
        token_embeddings: torch.Tensor,
        segment_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Add segment embeddings to token embeddings.

        Args:
            token_embeddings: Token embeddings of shape (batch, seq_len, hidden).
            segment_ids: Segment IDs of shape (batch, seq_len).

        Returns:
            Combined embeddings of shape (batch, seq_len, hidden).
        """
        return token_embeddings + self.segment_embedding(segment_ids)
