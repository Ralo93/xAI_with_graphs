import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from pathlib import Path


# Use forward slashes so it can run on windows and linux
MODEL_FILE_NAME = 'app/mlruns/10/ec80f1fe720946fe9f11126ec5831338/artifacts/model/data/model.pth'

class GAT(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, 
                 num_heads: int = 4, dropout: float = 0.3, edge_dim: int = None):
        super().__init__()
        
        self.dropout = nn.Dropout(dropout)
        
        # Calculate output dimensions
        self.gat1_out_channels = hidden_channels * num_heads
        self.gat2_out_channels = hidden_channels * num_heads
        
        # Replace BatchNorm with LayerNorm
        self.norm1 = nn.LayerNorm(self.gat1_out_channels)
        self.norm2 = nn.LayerNorm(self.gat2_out_channels)

        # Input projection for skip connection
        self.proj_skip = nn.Linear(in_channels, self.gat2_out_channels)
        
        self.gat1 = GATConv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            heads=num_heads,
            concat=True,
            dropout=dropout,
            edge_dim=edge_dim,
            add_self_loops=False  ## The underlaying model might have self loops!!
        )
        
        self.gat2 = GATConv(
            in_channels=hidden_channels * num_heads,
            out_channels=hidden_channels,
            heads=num_heads,
            concat=True,
            dropout=dropout,
            add_self_loops=False
        )
        
        self.gat3 = GATConv(
            in_channels=hidden_channels * num_heads,
            out_channels=out_channels,# // num_heads,
            heads=num_heads,
            concat=False,
            dropout=dropout,
            add_self_loops=False
        )

    def forward(self, x, edge_index):
        # Save input for skip connection
        x_skip = x

        # GAT layer 1
        x, alpha1 = self.gat1(x, edge_index, return_attention_weights=True)
        x = F.elu(x)
        x = self.norm1(x)
        x = self.dropout(x)

        # GAT layer 2
        x_skip = self.proj_skip(x_skip)  # Align dimensions for skip connection
        x = x + x_skip  # Add skip connection
        x, alpha2 = self.gat2(x, edge_index, return_attention_weights=True)
        x = F.elu(x)
        x = self.norm2(x)
        x = self.dropout(x)

        # GAT layer 3
        x, alpha3 = self.gat3(x, edge_index, return_attention_weights=True)

        return x, [alpha1, alpha2, alpha3]
    

def get_raw_model(in_channels: int, hidden_channels: int, out_channels: int, 
                  num_heads: int = 4, dropout: float = 0.3, edge_dim: int = None) -> GAT:
    """
    Create a model with the same architecture as the one used during training, 
    but without any weights
    """
    return GAT(
        in_channels=in_channels, 
        hidden_channels=hidden_channels, 
        out_channels=out_channels,
        num_heads=num_heads,
        dropout=dropout,
        edge_dim=edge_dim
    )

def load_model(
    in_channels: int, 
    hidden_channels: int, 
    out_channels: int,
    model_path: str = None,
    num_heads: int = 4, 
    dropout: float = 0.3, 
    edge_dim: int = None
) -> GAT:
    """
    Load the model with its trained weights
    
    Args:
        in_channels (int): Number of input features
        hidden_channels (int): Number of hidden channels
        out_channels (int): Number of output channels
        model_path (str, optional): Path to the model weights. 
                                    Defaults to models/model.pth
        num_heads (int, optional): Number of attention heads
        dropout (float, optional): Dropout rate
        edge_dim (int, optional): Dimension of edge features
    
    Returns:
        GAT: Loaded model with trained weights
    """
    # Use default path if not provided
    if model_path is None:
        model_path = MODEL_FILE_NAME
    
    # Create raw model with specified architecture
    model = get_raw_model(
        in_channels=in_channels, 
        hidden_channels=hidden_channels, 
        out_channels=out_channels,
        num_heads=num_heads,
        dropout=dropout,
        edge_dim=edge_dim
    )
    
    # Load the model weights
    try:
        model = torch.load(model_path, map_location='cpu')
        #model.eval()
    except Exception as e:
        print(f"Error loading model weights: {e}")
        raise
    
    # Set model to evaluation mode
    model.eval()
    
    return model

# Example usage
if __name__ == "__main__":
    # Example parameters - adjust these to match your model
    model = load_model(
        in_channels=3,      # Adjust to your node feature dimension
        hidden_channels=64, # Adjust to your model's hidden layer size
        out_channels=6,     # Number of output classes
        model_path=None     # Uses default path
    )
    print(model)