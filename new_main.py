import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from typing import Optional, Tuple
from torch import Tensor
from torch.nn import Linear
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from sklearn.model_selection import train_test_split
import torch.nn.functional as F


def load_cora(train_ratio=0.05, val_ratio=0.95):
    """
    Load and preprocess the Cora dataset for node classification.

    Args:
        train_ratio (float): Ratio of nodes for training.
        val_ratio (float): Ratio of nodes for validation.
    
    Returns:
        data: Preprocessed graph data object.
    """
    # Load Cora dataset
    dataset = Planetoid(root='data/Cora', name='Cora', transform=NormalizeFeatures())
    data = dataset[0]

    # Create train, val, and test masks
    num_nodes = data.num_nodes
    indices = torch.arange(num_nodes)

    train_size = int(train_ratio * num_nodes)
    val_size = int(val_ratio * num_nodes)

    # Shuffle and split indices
    train_indices, test_indices = train_test_split(indices, train_size=train_size, shuffle=True, random_state=42)
    val_indices, test_indices = train_test_split(test_indices, train_size=val_size, shuffle=True, random_state=42)

    # Create boolean masks
    data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    data.train_mask[train_indices] = True
    data.val_mask[val_indices] = True
    data.test_mask[test_indices] = True

    return data


class Config:
    """Centralized configuration for the CoGNN model"""
    class Action:
        # Action Network Configuration
        ACTIVATION = F.relu
        DROPOUT = 0.2
        HIDDEN_DIM = 64
        NUM_LAYERS = 2
        INPUT_DIM = 128
        OUTPUT_DIM = 2
        AGG = 'mean'

    class Environment:
        # Environment Network Configuration
        INPUT_DIM = 1433
        OUTPUT_DIM = 7
        NUM_LAYERS = 3
        DROPOUT = 0.2
        HIDDEN_DIM = 128
        LAYER_NORM = False
        SKIP_CONNECTION = True
        AGG = 'sum'

    class Gumbel:
        # Gumbel Softmax Configuration
        TEMPERATURE = 0.5
        TAU = 0.01
        LEARN_TEMPERATURE = False

    class Training:
        # Training Hyperparameters
        LEARNING_RATE = 0.001
        WEIGHT_DECAY = 0
        EPOCHS = 200
        BATCH_SIZE = 32
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class EncoderLinear(Linear):
    def forward(self, x: Tensor, pestat=None) -> Tensor:
        return super().forward(x)

class WeightedGNNConv(torch_geometric.nn.MessagePassing):
    def __init__(self, in_channels: int, out_channels: int, aggr='add', bias=True):
        super().__init__(aggr=aggr)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lin = nn.Linear(2 * in_channels, out_channels, bias=bias)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_attr: Optional[torch.Tensor] = None, 
                edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, edge_weight=edge_weight)
        out = self.lin(torch.cat((x, out), dim=-1))
        return out

    def message(self, x_j: torch.Tensor, edge_attr: Optional[torch.Tensor] = None, 
                edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Custom message passing logic
        msg = x_j if edge_attr is None else x_j * edge_attr
        if edge_weight is not None:
            msg = msg * edge_weight.view(-1, 1)
        return msg

class ActionNet(nn.Module):
    def __init__(self, config=Config.Action):
        super().__init__()
        self.num_layers = config.NUM_LAYERS
        self.dropout = nn.Dropout(config.DROPOUT)
        self.act = config.ACTIVATION

        # Dynamic network creation based on layers
        self.net = nn.ModuleList([
            WeightedGNNConv(
                config.INPUT_DIM if i == 0 else config.HIDDEN_DIM, 
                config.HIDDEN_DIM if i < config.NUM_LAYERS - 1 else config.OUTPUT_DIM,
                aggr=config.AGG  # Use aggregation method from config
            ) for i in range(config.NUM_LAYERS)
        ])

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                env_edge_attr: Optional[torch.Tensor] = None, 
                act_edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Prepare edge attributes for each layer
        edge_attrs = [env_edge_attr] + [act_edge_attr] * (self.num_layers - 1)
        
        for idx, (edge_attr, layer) in enumerate(zip(edge_attrs, self.net)):
            x = layer(x, edge_index, edge_attr)
            if idx < len(self.net) - 1:  # Don't apply dropout and activation on last layer
                x = self.dropout(x)
                x = self.act(x)
        
        return x
    
    
class CoGNN(nn.Module):
    def __init__(self, 
                 gumbel_args=Config.Gumbel, 
                 env_args=Config.Environment, 
                 action_args=Config.Action):
        super().__init__()
        
        # Environment network configuration
        self.env_net = nn.ModuleList([
            # First layer: EncoderLinear
            EncoderLinear(env_args.INPUT_DIM, env_args.HIDDEN_DIM)
        ] + [
            # Intermediate layers: WeightedGNNConv
            WeightedGNNConv(env_args.HIDDEN_DIM, env_args.HIDDEN_DIM) 
            for _ in range(env_args.NUM_LAYERS - 1)
        ] + [
            # Final layer: Linear decoder
            nn.Linear(env_args.HIDDEN_DIM, env_args.OUTPUT_DIM)
        ])

        # Layer normalization (or Identity)
        self.hidden_layer_norm = nn.Identity() if not env_args.LAYER_NORM else nn.LayerNorm(env_args.HIDDEN_DIM)
        
        # Dropout
        self.dropout = nn.Dropout(p=env_args.DROPOUT)
        
        # Gumbel softmax configuration
        self.learn_temp = gumbel_args.LEARN_TEMPERATURE
        self.temp = gumbel_args.TEMPERATURE

        # Action networks
        self.in_act_net = ActionNet(action_args)
        self.out_act_net = ActionNet(action_args)

        # Pooling function
        #self.pooling = pool()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, pestat=None, 
                edge_attr: Optional[torch.Tensor] = None, 
                batch: Optional[torch.Tensor] = None,
                edge_ratio_node_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # Initial node encoding
        x = self.env_net[0](x, pestat)
        x = self.dropout(x)

        # Process through GNN layers
        for layer_idx in range(1, len(self.env_net) - 1):
            # Normalize hidden states
            x = self.hidden_layer_norm(x)

            # Action logits
            in_logits = self.in_act_net(x, edge_index)
            out_logits = self.out_act_net(x, edge_index)

            # Gumbel Softmax
            in_probs = F.gumbel_softmax(logits=in_logits, tau=self.temp, hard=True)
            out_probs = F.gumbel_softmax(logits=out_logits, tau=self.temp, hard=True)

            # Create edge weights
            edge_weight = self.create_edge_weight(edge_index, in_probs[:, 0], out_probs[:, 0])

            # Apply graph convolution
            x = self.env_net[layer_idx](x, edge_index, edge_weight=edge_weight)
            x = self.dropout(x)

        # Final layer (decoder)
        x = self.env_net[-1](x)

        edge_ratio_tensor = -1 * torch.ones(size=(len(self.env_net)-2,), device=x.device)

        return x, edge_ratio_tensor

    def create_edge_weight(self, edge_index: torch.Tensor, 
                            keep_in_prob: torch.Tensor, 
                            keep_out_prob: torch.Tensor) -> torch.Tensor:
        u, v = edge_index
        edge_in_prob = keep_in_prob[v]
        edge_out_prob = keep_out_prob[u]
        return edge_in_prob * edge_out_prob
    
def train_cognn(data, config=Config.Training):
    """
    Training function for the CoGNN model with Cora dataset.
    
    Args:
        data: PyTorch Geometric data object with train/val/test masks.
        config: Training configuration.
    
    Returns:
        Trained model
    """
    data = data.to(config.DEVICE)

    # Initialize model, optimizer, and loss
    model = CoGNN().to(config.DEVICE)
    optimizer = Adam(
        model.parameters(), 
        lr=config.LEARNING_RATE, 
        weight_decay=config.WEIGHT_DECAY
    )
    criterion = CrossEntropyLoss()

    for epoch in range(config.EPOCHS):
        model.train()
        optimizer.zero_grad()

        # Forward pass
        out, _ = model(data.x, data.edge_index)
        train_loss = criterion(out[data.train_mask], data.y[data.train_mask])
        train_loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = criterion(out[data.val_mask], data.y[data.val_mask])
            val_pred = out[data.val_mask].argmax(dim=1)
            val_accuracy = (val_pred == data.y[data.val_mask]).sum().item() / data.val_mask.sum().item()

        print(f'Epoch {epoch + 1}/{config.EPOCHS}:')
        print(f'Train Loss: {train_loss.item():.4f}')
        print(f'Val Loss: {val_loss.item():.4f}, Val Accuracy: {val_accuracy:.4f}')

    return model


# Example usage
if __name__ == "__main__":
    # Create toy dataset
    cora_data = load_cora()
    
    # Train the model
    trained_model = train_cognn(cora_data)