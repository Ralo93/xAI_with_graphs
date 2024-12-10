import sys
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
import torch.nn.functional as F
import tqdm
import networkx as nx
from pyvis.network import Network

def load_cora(train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
    """
    Load and preprocess the Cora dataset for node classification.

    Args:
        train_ratio (float): Ratio of nodes for training.
        val_ratio (float): Ratio of nodes for validation.
        test_ratio (float): Ratio of nodes for testing.
    
    Returns:
        data: Preprocessed graph data object.
    """
    assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must sum up to 1."

    # Load Cora dataset
    dataset = Planetoid(root='data/Cora', name='Cora', transform=NormalizeFeatures())
    data = dataset[0]

    # Create train, val, and test masks
    num_nodes = data.num_nodes
    indices = torch.arange(num_nodes)

    train_size = int(train_ratio * num_nodes)
    val_size = int(val_ratio * num_nodes)
    test_size = num_nodes - train_size - val_size  # Ensure all nodes are accounted for

    # Shuffle and split indices
    shuffled_indices = indices[torch.randperm(num_nodes)]
    train_indices = shuffled_indices[:train_size]
    val_indices = shuffled_indices[train_size:train_size + val_size]
    test_indices = shuffled_indices[train_size + val_size:]

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
        ACTIVATION = F.relu # could be also F.relu but gelu has way better performance
        DROPOUT = 0.5
        HIDDEN_DIM = 32 # independent
        NUM_LAYERS = 1
        INPUT_DIM = 32 # needs to be the same as hidden dimension in environment network
        OUTPUT_DIM = 2
        AGG = 'sum'

    class Environment:
        # Environment Network Configuration
        INPUT_DIM = 1433
        OUTPUT_DIM = 7
        NUM_LAYERS = 3 #3
        DROPOUT = 0.5
        HIDDEN_DIM = 32
        LAYER_NORM = False
        SKIP_CONNECTION = True
        AGG = 'sum'

    class Gumbel:
        # Gumbel Softmax Configuration
        TEMPERATURE = 0.5
        TAU = 0.01
        LEARN_TEMPERATURE = True

    class Training:
        # Training Hyperparameters
        LEARNING_RATE = 0.001
        WEIGHT_DECAY = 0
        EPOCHS = 500
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
    
class GraphLinear(Linear):
    def forward(self, x: Tensor, edge_index: Optional[Tensor] = None) -> Tensor:
        # Ignore edge_index as Linear works on node features only
        return super().forward(x)


class TempSoftPlus(nn.Module):
    def __init__(self, gumbel_config=Config.Gumbel, env_dim: int = Config.Environment.HIDDEN_DIM):
        super(TempSoftPlus, self).__init__()
        self.linear_model = GraphLinear(env_dim, 1, bias=False)  # Simple linear model for demonstration
        self.tau0 = gumbel_config.TAU
        self.learn_temp = gumbel_config.LEARN_TEMPERATURE
        self.softplus = nn.Softplus(beta=1)

    def forward(self, x: Tensor, edge_index: Optional[Tensor] = None) -> Tensor:
        # Pass only x to linear_model, as edge_index is unused here
        x = self.linear_model(x)
        x = self.softplus(x) + self.tau0
        temp = x.pow(-1)
        return temp.masked_fill(temp == float('inf'), 0.0)


    
class CoGNN(nn.Module):
    def __init__(self, 
                 gumbel_args=Config.Gumbel, 
                 env_args=Config.Environment, 
                 action_args=Config.Action,
                 ):
        super().__init__()

        # Gumbel softmax configuration
        self.learn_temp = gumbel_args.LEARN_TEMPERATURE
        self.temp = gumbel_args.TEMPERATURE

        if self.learn_temp:
            self.temp_model = TempSoftPlus()
        
        # Environment network configuration
        self.env_net = nn.ModuleList([
            # First layer: EncoderLinear
            EncoderLinear(env_args.INPUT_DIM, env_args.HIDDEN_DIM)
        ] + [
            # Intermediate layers: WeightedGNNConv
            WeightedGNNConv(env_args.HIDDEN_DIM, env_args.HIDDEN_DIM) 
            for _ in range(env_args.NUM_LAYERS)
        ] + [
            # Final layer: Linear decoder
            nn.Linear(env_args.HIDDEN_DIM, env_args.OUTPUT_DIM)
        ])

        # Layer normalization (or Identity)
        self.hidden_layer_norm = nn.Identity() if not env_args.LAYER_NORM else nn.LayerNorm(env_args.HIDDEN_DIM)
        
        # Dropout
        self.dropout = nn.Dropout(p=env_args.DROPOUT)
        

        # Action networks
        self.in_act_net = ActionNet(action_args)
        self.out_act_net = ActionNet(action_args)

        self.edge_weights_by_layer = []  # To store edge weights for each layer


    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, pestat=None, 
                edge_attr: Optional[torch.Tensor] = None, 
                batch: Optional[torch.Tensor] = None,
                edge_ratio_node_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # Initial node encoding
        x = self.env_net[0](x, pestat)
        x = self.dropout(x)

        self.edge_weights_by_layer = []  # Reset weights for this forward pass

        # Process through GNN layers
        for layer_idx in range(1, len(self.env_net) - 1):
            # Normalize hidden states
            x = self.hidden_layer_norm(x)

            # Action logits
            in_logits = self.in_act_net(x, edge_index)
            out_logits = self.out_act_net(x, edge_index)

            temp = self.temp_model(x=x, edge_index=edge_index) if self.learn_temp else self.temp
            #print(temp)

            # Gumbel Softmax
            in_probs = F.gumbel_softmax(logits=in_logits, tau=temp, hard=True)
            out_probs = F.gumbel_softmax(logits=out_logits, tau=temp, hard=True)

            print(f"in_probs: {in_probs}")
            print(f"out_probs: {out_probs}")

            # Create edge weights
            edge_weight = self.create_edge_weight(edge_index, in_probs[:, 0], out_probs[:, 0])
            self.edge_weights_by_layer.append(edge_weight) 

            print(f"Edge weights in layer {layer_idx}: {self.edge_weights_by_layer[-1]}")

            assert torch.all(torch.logical_or(edge_weight == 0, edge_weight == 1)), \
                "Edge weights must be either 0 or 1"

            # Apply graph convolution
            x = self.env_net[layer_idx](x, edge_index, edge_weight=edge_weight)
            x = self.dropout(x)

        # Final layer (decoder)
        x = self.env_net[-1](x)

            # Final validation
        for i, ew in enumerate(self.edge_weights_by_layer):
            assert torch.all(torch.logical_or(ew == 0, ew == 1)), f"In Model Edge weights in layer {i} are not binary at the end"

        for i, ew in enumerate(self.edge_weights_by_layer):
            print(f"Edge weights before return, layer {i}: {ew}")

        return x, self.edge_weights_by_layer

    def create_edge_weight(self, edge_index: torch.Tensor, 
                            keep_in_prob: torch.Tensor, 
                            keep_out_prob: torch.Tensor) -> torch.Tensor:
        u, v = edge_index
        edge_in_prob = keep_in_prob[v]
        edge_out_prob = keep_out_prob[u]

        edge_weight = edge_in_prob * edge_out_prob

            # Assert that edge_weight is either 0 or 1
        assert torch.all(torch.logical_or(edge_weight == 0, edge_weight == 1)), \
            "Edge weights must be either 0 or 1"

        return edge_weight
    

def train_cognn(data, config=Config.Training, model_path='cognn_model.pth'):
    """
    Training function for the CoGNN model with Cora dataset.
    
    Args:
        data: PyTorch Geometric data object with train/val/test masks.
        config: Training configuration.
        model_path: Path to save the trained model.
    
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

    with tqdm.tqdm(total=config.EPOCHS, file=sys.stdout, desc="Training Progress") as pbar:

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

            # Update progress bar
            pbar.set_postfix({
                "Train Loss": f"{train_loss.item():.4f}",
                "Val Loss": f"{val_loss.item():.4f}",
                "Val Accuracy": f"{val_accuracy:.4f}"
            })
            pbar.update(1)

    # Evaluate on the test set
    model.eval()
    with torch.no_grad():
        out, _ = model(data.x, data.edge_index)
        test_pred = out[data.test_mask].argmax(dim=1)
        test_accuracy = (test_pred == data.y[data.test_mask]).sum().item() / data.test_mask.sum().item()
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Extract and save learned edge weights
    with torch.no_grad():
        model.eval()
        model(data.x, data.edge_index)  # Forward pass to populate edge weights
        #save_learned_edge_weights(model, data.edge_index)

    # Save the entire model
    torch.save({
        'model_state_dict': model.state_dict()
    }, model_path)
    print(f"Model saved to {model_path}")

    return model



def save_learned_edge_weights(model, edge_index, filename_prefix="edge_weights_layer"):
    """
    Save the learned edge weights for all layers.

    Args:
        model (CoGNN): Trained CoGNN model.
        edge_index (Tensor): Edge indices of the graph.
        filename_prefix (str): Prefix for the output files.
    """
    u, v = edge_index.cpu().numpy()

    for layer_idx, edge_weights in enumerate(model.edge_weights_by_layer):
        edge_weights = edge_weights.cpu().numpy()

        # Create a weighted directed graph
        G = nx.DiGraph()
        for i in range(len(u)):
            G.add_edge(int(u[i]), int(v[i]), weight=float(edge_weights[i]))  # Convert weight to Python float

        # Visualize using PyVis
        net = Network(notebook=False, directed=True)

        # Add all nodes explicitly to avoid missing nodes
        all_nodes = set(map(int, u)).union(set(map(int, v)))  # Ensure all nodes are Python integers
        for node in all_nodes:
            net.add_node(node, label=str(node))

        # Add edges with weights
        for src, tgt, data in G.edges(data=True):
            net.add_edge(
                src, tgt,
                title=f"Weight: {data['weight']:.4f}",
                value=float(data['weight'])  # Convert weight to Python float
            )

        # Save the graph as an HTML file
        net.write_html(f"{filename_prefix}_{layer_idx}.html")



if __name__ == "__main__":

    torch.manual_seed(3)
    # Specify ratios for train, validation, and test sets
    cora_data = load_cora(train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)

    # Train the model
    trained_model = train_cognn(cora_data)

    # Save graphs for input and output action networks
    edge_index = cora_data.edge_index


