import os
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from pathlib import Path
from typing import Optional, Tuple
from torch import Tensor
from torch.nn import Linear
import torch_geometric


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
    def __init__(self, config):
        super().__init__()
        self.num_layers = config.NUM_LAYERS
        
        self.act = config.ACTIVATION

        # Dynamic network creation based on layers
        self.net = nn.ModuleList([
            WeightedGNNConv(
                config.INPUT_DIM if i == 0 else config.HIDDEN_DIM, 
                config.HIDDEN_DIM if i < config.NUM_LAYERS - 1 else config.OUTPUT_DIM,
                aggr=config.AGG  # Use aggregation method from config
            ) for i in range(config.NUM_LAYERS)
        ])
        self.dropout = nn.Dropout(config.DROPOUT)

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
    def __init__(self, gumbel_config, env_dim):
        super(TempSoftPlus, self).__init__()
        self.linear_model = GraphLinear(env_dim, 1, bias=False)
        self.linear_model = nn.ModuleList([GraphLinear(env_dim, 1, bias=False) for _ in range(1)])
        self.tau0 = gumbel_config.TAU
        self.learn_temp = gumbel_config.LEARN_TEMPERATURE
        self.softplus = nn.Softplus(beta=1)

    def forward(self, x: torch.Tensor, edge_index: Optional[Tensor] = None) -> Tensor:
        # Pass only x to linear_model, as edge_index is unused here
        for layer in self.linear_model:
            x = layer(x)
        x = self.softplus(x) + self.tau0
        temp = x.pow(-1)
        return temp.masked_fill(temp == float('inf'), 0.0)


    
class CoGNN(nn.Module):
    def __init__(self, 
                 gumbel_args, 
                 env_args, 
                 action_args,
                 ):
        super().__init__()

        print(gumbel_args[0])
        gumbel_args = gumbel_args[0]
        env_args = env_args[0]
        #action_args = action_args[0]

        # Gumbel softmax configuration
        self.learn_temp = gumbel_args.LEARN_TEMPERATURE
        self.temp = gumbel_args.TEMPERATURE

        if self.learn_temp:
            self.temp_model = TempSoftPlus(gumbel_args, env_args.HIDDEN_DIM)
        
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

            temp = self.temp_model(x, edge_index) if self.learn_temp else self.temp
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

        print(edge_weight)

            # Assert that edge_weight is either 0 or 1
        assert torch.all(torch.logical_or(edge_weight == 0, edge_weight == 1)), \
            "Edge weights must be either 0 or 1"
        print(f"Edge weights created (binary check): {edge_weight}")

        return edge_weight
    
# Use forward slashes so it can run on windows and linux
MODEL_FILE_NAME = 'app/mlruns/10/ec80f1fe720946fe9f11126ec5831338/artifacts/model/data/model.pth'

class GAT(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, 
                 num_heads: int = 5, dropout: float = 0.3, edge_dim: int = None):
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
        x, alpha2 = self.gat2(x, edge_index, return_attention_weights=True)
        x = F.elu(x)
        x = self.norm2(x)
        x = self.dropout(x)

        # GAT layer 3
        x, alpha3 = self.gat3(x, edge_index, return_attention_weights=True)

        # Skip connection to output
        x_skip = self.proj_skip(x_skip)  # Align dimensions for skip connection if needed
        x = x + x_skip  # Add skip connection to final output

        return x, [alpha1, alpha2, alpha3]

    

def get_raw_model(in_channels: int, hidden_channels: int, out_channels: int, 
                  num_heads: int = 5, dropout: float = 0.3, edge_dim: int = None) -> GAT:
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

def load_model_3layerGat(
    in_channels: int, 
    hidden_channels: int, 
    out_channels: int,
    model_path: str = None,
    num_heads: int = 5, 
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
        model.eval()
    except Exception as e:
        print(f"Error loading model weights: {e}")
        raise
    
    # Set model to evaluation mode
    model.eval()
    
    return model


def get_raw_cognn_model(
    gumbel_args, 
    env_args, 
    action_args
) -> CoGNN:
    """
    Create a CoGNN model with the same architecture as the one used during training, 
    but without any weights
    
    Args:
        gumbel_args: Gumbel softmax configuration
        env_args: Environment network configuration
        action_args: Action network configuration
    
    Returns:
        CoGNN: Uninitialized model with specified architecture
    """
    return CoGNN(
        gumbel_args=gumbel_args, 
        env_args=env_args, 
        action_args=action_args
    )

def load_cognn_model_cora_3layer(
    config,
    model_path: str = 'app/cognn_model_cora_3layer.pth',
) -> CoGNN:
    """
    Load the CoGNN model with its trained weights
    
    Args:
        model_path (str): Path to the saved model checkpoint
        gumbel_args: Gumbel softmax configuration
        env_args: Environment network configuration
        action_args: Action network configuration
    
    Returns:
        CoGNN: Loaded model with trained weights
    """
    gumbel_args=config.Gumbel, 
    env_args=config.Environment, 
    action_args=config.Action

    for con in gumbel_args:
        print(con)
    
    # Create raw model with specified architecture
    model = get_raw_cognn_model(
        gumbel_args=gumbel_args, 
        env_args=env_args, 
        action_args=action_args
    )

    print("Model state dictionary keys:")
    for key in model.state_dict().keys():
        print(key)
    
    # Load the model checkpoint
    try:
        #checkpoint = torch.load(model_path, map_location='cpu')
        
        # Load the model state dictionary
        #model.load_state_dict(checkpoint['model_state_dict']) #TODO change back 
                # Load the checkpoint

        # for my checkpoints:
        checkpoint = torch.load(model_path)
        #checkpoint = torch.load(f'best_model_fold_{num_fold}.pth')

        # Extract the model state_dict
        model.load_state_dict(checkpoint['model_state_dict'])

        #model.to(device)
        
        # Optional: print out additional information from the checkpoint
        print(f"Loaded coGNN model")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        raise
    
    # Set model to evaluation mode
    model.eval()
    
    return model


def load_cognn_model_cora_5layer(
    config,
    model_path: str = 'app/cognn_model_cora_5layer.pth',
) -> CoGNN:
    """
    Load the CoGNN model with its trained weights
    
    Args:
        model_path (str): Path to the saved model checkpoint
        gumbel_args: Gumbel softmax configuration
        env_args: Environment network configuration
        action_args: Action network configuration
    
    Returns:
        CoGNN: Loaded model with trained weights
    """
    gumbel_args=config.Gumbel, 
    env_args=config.Environment, 
    action_args=config.Action

    for con in gumbel_args:
        print(con)
    
    # Create raw model with specified architecture
    model = get_raw_cognn_model(
        gumbel_args=gumbel_args, 
        env_args=env_args, 
        action_args=action_args
    )

    print("Model state dictionary keys:")
    for key in model.state_dict().keys():
        print(key)
    
    # Load the model checkpoint
    try:
        #checkpoint = torch.load(model_path, map_location='cpu')
        
        # Load the model state dictionary
        #model.load_state_dict(checkpoint['model_state_dict']) #TODO change back 
                # Load the checkpoint

        # for my checkpoints:
        checkpoint = torch.load(model_path)
        #checkpoint = torch.load(f'best_model_fold_{num_fold}.pth')

        # Extract the model state_dict
        model.load_state_dict(checkpoint['model_state_dict'])

        #model.to(device)
        
        # Optional: print out additional information from the checkpoint
        print(f"Loaded coGNN model")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        raise
    
    # Set model to evaluation mode
    model.eval()
    
    return model


def load_cognn_model_re_3layer(
    config,
    model_path: str = 'app/cognn_model_re_3layer.pth',
) -> CoGNN:
    """
    Load the CoGNN model with its trained weights
    
    Args:
        model_path (str): Path to the saved model checkpoint
        gumbel_args: Gumbel softmax configuration
        env_args: Environment network configuration
        action_args: Action network configuration
    
    Returns:
        CoGNN: Loaded model with trained weights
    """
    gumbel_args=config.Gumbel, 
    env_args=config.Environment, 
    action_args=config.Action

    for con in gumbel_args:
        print(con)
    
    # Create raw model with specified architecture
    model = get_raw_cognn_model(
        gumbel_args=gumbel_args, 
        env_args=env_args, 
        action_args=action_args
    )

    print("Model state dictionary keys:")
    for key in model.state_dict().keys():
        print(key)
    
    # Load the model checkpoint
    try:
        #checkpoint = torch.load(model_path, map_location='cpu')
        
        # Load the model state dictionary
        #model.load_state_dict(checkpoint['model_state_dict']) #TODO change back 
                # Load the checkpoint

        # for my checkpoints:
        checkpoint = torch.load(model_path)
        #checkpoint = torch.load(f'best_model_fold_{num_fold}.pth')

        # Extract the model state_dict
        model.load_state_dict(checkpoint['model_state_dict'])

        #model.to(device)
        
        # Optional: print out additional information from the checkpoint
        print(f"Loaded coGNN model")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        raise
    
    # Set model to evaluation mode
    model.eval()
    
    return model


def load_cognn_model_re_5layer(
    config,
    model_path: str = 'app/cognn_model_re_5layer.pth',
) -> CoGNN:
    """
    Load the CoGNN model with its trained weights
    
    Args:
        model_path (str): Path to the saved model checkpoint
        gumbel_args: Gumbel softmax configuration
        env_args: Environment network configuration
        action_args: Action network configuration
    
    Returns:
        CoGNN: Loaded model with trained weights
    """
    gumbel_args=config.Gumbel, 
    env_args=config.Environment, 
    action_args=config.Action

    print(env_args)

    for con in gumbel_args:
        print(con)
    
    # Create raw model with specified architecture
    model = get_raw_cognn_model(
        gumbel_args=gumbel_args, 
        env_args=env_args, 
        action_args=action_args
    )

    print("Model state dictionary keys:")
    for key in model.state_dict().keys():
        print(key)
    
    # Load the model checkpoint
    try:
        #checkpoint = torch.load(model_path, map_location='cpu')
        
        # Load the model state dictionary
        #model.load_state_dict(checkpoint['model_state_dict']) #TODO change back 
                # Load the checkpoint

        # for my checkpoints:
        checkpoint = torch.load(model_path)
        #checkpoint = torch.load(f'best_model_fold_{num_fold}.pth')

        # Extract the model state_dict
        model.load_state_dict(checkpoint['model_state_dict'])

        #model.to(device)
        
        # Optional: print out additional information from the checkpoint
        print(f"Loaded coGNN model")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        raise
    
    # Set model to evaluation mode
    model.eval()
    
    return model