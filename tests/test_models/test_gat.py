import pytest
import torch
from src.models.gat import GAT, GATConfig


class TestGAT:
    def test_forward_pass(self, sample_model, cora_dataset):
        """Test forward pass of GAT model"""
        # Prepare input
        x = cora_dataset.node_features
        edge_index = cora_dataset.edges

        # Run forward pass
        output = sample_model(x, edge_index)

        # Check output shape
        assert output.shape[0] == x.shape[0]  # Same number of nodes
        assert output.shape[1] == 7  # Number of Cora classes

    def test_attention_weights(self, sample_model, cora_dataset):
        """Test attention weight computation"""
        x = cora_dataset.node_features
        edge_index = cora_dataset.edges

        # Get attention weights
        output, attention_weights = sample_model(
            x, edge_index, return_attention_weights=True
        )

        # Check attention weights structure
        assert len(attention_weights) == 3  # Three GAT layers
        for layer_weights in attention_weights:
            assert len(layer_weights) == 2  # Edge index and weights
            edge_index, weights = layer_weights
            assert edge_index.shape[0] == 2  # Source and target nodes
            assert weights.shape[1] == 4  # Number of attention heads
