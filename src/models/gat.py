# src/models/gat.py
from dataclasses import dataclass
from typing import Optional, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


@dataclass
class GATConfig:
    """Configuration for GAT model"""

    in_channels: int
    hidden_channels: int
    out_channels: int
    num_heads: int = 5
    dropout: float = 0.3
    edge_dim: Optional[int] = None


class GAT(torch.nn.Module):
    def __init__(self, config: GATConfig):
        super().__init__()

        self.dropout = nn.Dropout(config.dropout)

        # Calculate output dimensions
        self.gat1_out_channels = config.hidden_channels * config.num_heads
        self.gat2_out_channels = config.hidden_channels * config.num_heads

        # Layer normalization
        self.norm1 = nn.LayerNorm(self.gat1_out_channels)
        self.norm2 = nn.LayerNorm(self.gat2_out_channels)

        # Input projection for skip connection
        self.proj_skip = nn.Linear(config.in_channels, self.gat2_out_channels)

        # GAT layers
        self.gat1 = GATConv(
            in_channels=config.in_channels,
            out_channels=config.hidden_channels,
            heads=config.num_heads,
            concat=True,
            dropout=config.dropout,
            edge_dim=config.edge_dim,
            add_self_loops=False,
        )

        self.gat2 = GATConv(
            in_channels=config.hidden_channels * config.num_heads,
            out_channels=config.hidden_channels,
            heads=config.num_heads,
            concat=True,
            dropout=config.dropout,
            add_self_loops=False,
        )

        self.gat3 = GATConv(
            in_channels=config.hidden_channels * config.num_heads,
            out_channels=config.out_channels,
            heads=config.num_heads,
            concat=False,
            dropout=config.dropout,
            add_self_loops=False,
        )

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> Tuple[torch.Tensor, List]:
        """
        Forward pass through the GAT model

        Args:
            x: Node features tensor
            edge_index: Edge index tensor

        Returns:
            tuple: (Output tensor, List of attention weights)
        """
        # Save input for skip connection
        x_skip = x

        # GAT layer 1
        x, alpha1 = self.gat1(x, edge_index, return_attention_weights=True)
        x = F.elu(x)
        x = self.norm1(x)
        x = self.dropout(x)

        # GAT layer 2 with skip connection
        x_skip = self.proj_skip(x_skip)
        x = x + x_skip
        x, alpha2 = self.gat2(x, edge_index, return_attention_weights=True)
        x = F.elu(x)
        x = self.norm2(x)
        x = self.dropout(x)

        # GAT layer 3
        x, alpha3 = self.gat3(x, edge_index, return_attention_weights=True)

        return x, [alpha1, alpha2, alpha3]
