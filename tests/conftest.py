import pytest
import torch
from src.data.cora import CoraDataset
from src.models.gat import GAT, GATConfig


@pytest.fixture
def cora_dataset():
    """Fixture for loading Cora dataset"""
    return CoraDataset("data/cora.npz")


@pytest.fixture
def sample_model():
    """Fixture for creating a sample GAT model"""
    config = GATConfig(
        in_channels=1433,  # Cora features
        hidden_channels=8,
        out_channels=7,  # Cora classes
        num_heads=4,
        dropout=0.3,
    )
    return GAT(config)
