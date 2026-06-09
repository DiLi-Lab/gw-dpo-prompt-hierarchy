"""Tests for Instructional Segment Embeddings."""

import torch

from src.model.ise import InstructionalSegmentEmbedding


def test_normal_initialization():
    ise = InstructionalSegmentEmbedding(num_segments=6, hidden_size=64, init_std=0.01)
    # Weights should be near zero (normal with std=0.01) but not exactly zero
    assert ise.segment_embedding.weight.abs().max() < 0.5  # very generous bound
    assert ise.segment_embedding.weight.std() > 0.0  # not all zeros


def test_output_shape():
    ise = InstructionalSegmentEmbedding(num_segments=6, hidden_size=64)
    segment_ids = torch.tensor([[0, 1, 2, 3, 4, 5]])
    embeddings = ise(segment_ids)
    assert embeddings.shape == (1, 6, 64)


def test_additive_behavior():
    ise = InstructionalSegmentEmbedding(num_segments=6, hidden_size=64)
    with torch.no_grad():
        # Set segment 0 to ones, segment 5 to zeros for predictable test
        ise.segment_embedding.weight[0] = torch.ones(64)
        ise.segment_embedding.weight[5] = torch.zeros(64)

    token_embeds = torch.zeros(1, 3, 64)
    segment_ids = torch.tensor([[0, 5, 0]])

    result = ise.add_to_embeddings(token_embeds, segment_ids)
    assert torch.allclose(result[0, 0], torch.ones(64))
    assert torch.allclose(result[0, 1], torch.zeros(64))
    assert torch.allclose(result[0, 2], torch.ones(64))


def test_different_segments_get_different_embeddings():
    ise = InstructionalSegmentEmbedding(num_segments=6, hidden_size=64)
    with torch.no_grad():
        for i in range(6):
            ise.segment_embedding.weight[i] = torch.full((64,), float(i))

    segment_ids = torch.tensor([[0, 1, 2]])
    embeddings = ise(segment_ids)
    assert not torch.allclose(embeddings[0, 0], embeddings[0, 1])
    assert not torch.allclose(embeddings[0, 1], embeddings[0, 2])


def test_batch_handling():
    ise = InstructionalSegmentEmbedding(num_segments=6, hidden_size=64)
    segment_ids = torch.tensor([[0, 1, 2], [3, 4, 5]])
    embeddings = ise(segment_ids)
    assert embeddings.shape == (2, 3, 64)
