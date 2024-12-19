import os
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from pathlib import Path
from torch import Tensor
from torch.nn import Linear
import torch_geometric

# Consolidated reusable components
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
        msg = x_j if edge_attr is None else x_j * edge_attr
        if edge_weight is not None:
            msg = msg * edge_weight.view(-1, 1)
        return msg

class ActionNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_layers = config.NUM_LAYERS
        self.act = config.ACTIVATION
        self.net = nn.ModuleList([
            WeightedGNNConv(
                config.INPUT_DIM if i == 0 else config.HIDDEN_DIM, 
                config.HIDDEN_DIM if i < config.NUM_LAYERS - 1 else config.OUTPUT_DIM,
                aggr=config.AGG
            ) for i in range(config.NUM_LAYERS)
        ])
        self.dropout = nn.Dropout(config.DROPOUT)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                env_edge_attr: Optional[torch.Tensor] = None, 
                act_edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        edge_attrs = [env_edge_attr] + [act_edge_attr] * (self.num_layers - 1)
        for idx, (edge_attr, layer) in enumerate(zip(edge_attrs, self.net)):
            x = layer(x, edge_index, edge_attr)
            if idx < len(self.net) - 1:
                x = self.dropout(x)
                x = self.act(x)
        return x

class GraphLinear(Linear):
    def forward(self, x: Tensor, edge_index: Optional[Tensor] = None) -> Tensor:
        return super().forward(x)

class TempSoftPlus(nn.Module):
    def __init__(self, gumbel_config, env_dim):
        super().__init__()
        self.linear_model = nn.ModuleList([GraphLinear(env_dim, 1, bias=False) for _ in range(1)])
        self.tau0 = gumbel_config.TAU
        self.learn_temp = gumbel_config.LEARN_TEMPERATURE
        self.softplus = nn.Softplus(beta=1)

    def forward(self, x: torch.Tensor, edge_index: Optional[Tensor] = None) -> Tensor:
        for layer in self.linear_model:
            x = layer(x)
        x = self.softplus(x) + self.tau0
        temp = x.pow(-1)
        return temp.masked_fill(temp == float('inf'), 0.0)

class CoGNN(nn.Module):
    def __init__(self, gumbel_args, env_args, action_args):
        super().__init__()
        self.learn_temp = gumbel_args.LEARN_TEMPERATURE
        self.temp = gumbel_args.TEMPERATURE
        if self.learn_temp:
            self.temp_model = TempSoftPlus(gumbel_args, env_args.HIDDEN_DIM)

        self.env_net = nn.ModuleList([
            EncoderLinear(env_args.INPUT_DIM, env_args.HIDDEN_DIM)
        ] + [
            WeightedGNNConv(env_args.HIDDEN_DIM, env_args.HIDDEN_DIM) 
            for _ in range(env_args.NUM_LAYERS)
        ] + [
            nn.Linear(env_args.HIDDEN_DIM, env_args.OUTPUT_DIM)
        ])

        self.hidden_layer_norm = nn.Identity() if not env_args.LAYER_NORM else nn.LayerNorm(env_args.HIDDEN_DIM)
        self.dropout = nn.Dropout(p=env_args.DROPOUT)

        self.in_act_net = ActionNet(action_args)
        self.out_act_net = ActionNet(action_args)
        self.edge_weights_by_layer = []

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, pestat=None, 
                edge_attr: Optional[torch.Tensor] = None, 
                batch: Optional[torch.Tensor] = None,
                edge_ratio_node_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.env_net[0](x, pestat)
        x = self.dropout(x)
        self.edge_weights_by_layer = []

        for layer_idx in range(1, len(self.env_net) - 1):
            x = self.hidden_layer_norm(x)

            in_logits = self.in_act_net(x, edge_index)
            out_logits = self.out_act_net(x, edge_index)

            temp = self.temp_model(x, edge_index) if self.learn_temp else self.temp

            in_probs = F.gumbel_softmax(logits=in_logits, tau=temp, hard=True)
            out_probs = F.gumbel_softmax(logits=out_logits, tau=temp, hard=True)

            edge_weight = self.create_edge_weight(edge_index, in_probs[:, 0], out_probs[:, 0])
            self.edge_weights_by_layer.append(edge_weight)

            assert torch.all(torch.logical_or(edge_weight == 0, edge_weight == 1)), \
                "Edge weights must be either 0 or 1"

            x = self.env_net[layer_idx](x, edge_index, edge_weight=edge_weight)
            x = self.dropout(x)

        x = self.env_net[-1](x)
        return x, self.edge_weights_by_layer

    def create_edge_weight(self, edge_index: torch.Tensor, 
                           keep_in_prob: torch.Tensor, 
                           keep_out_prob: torch.Tensor) -> torch.Tensor:
        u, v = edge_index
        edge_in_prob = keep_in_prob[v]
        edge_out_prob = keep_out_prob[u]
        edge_weight = edge_in_prob * edge_out_prob
        return edge_weight

# GAT and CoGNN model handling
class GAT(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, 
                 num_heads: int = 5, dropout: float = 0.3, edge_dim: int = None):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.gat1_out_channels = hidden_channels * num_heads
        self.gat2_out_channels = hidden_channels * num_heads
        self.norm1 = nn.LayerNorm(self.gat1_out_channels)
        self.norm2 = nn.LayerNorm(self.gat2_out_channels)
        self.proj_skip = nn.Linear(in_channels, self.gat2_out_channels)
        self.gat1 = GATConv(in_channels, hidden_channels, heads=num_heads, concat=True, dropout=dropout, edge_dim=edge_dim)
        self.gat2 = GATConv(hidden_channels * num_heads, hidden_channels, heads=num_heads, concat=True, dropout=dropout)
        self.gat3 = GATConv(hidden_channels * num_heads, out_channels, heads=num_heads, concat=False, dropout=dropout)

    def forward(self, x, edge_index):
        x_skip = x
        x, alpha1 = self.gat1(x, edge_index, return_attention_weights=True)
        x = F.elu(self.norm1(self.dropout(x)))

        x, alpha2 = self.gat2(x, edge_index, return_attention_weights=True)
        x = F.elu(self.norm2(self.dropout(x)))

        x, alpha3 = self.gat3(x, edge_index, return_attention_weights=True)
        x_skip = self.proj_skip(x_skip)
        x = x + x_skip
        return x, [alpha1, alpha2, alpha3]

# Model loading utilities
def get_raw_model(in_channels: int, hidden_channels: int, out_channels: int, 
                  num_heads: int = 5, dropout: float = 0.3, edge_dim: int = None) -> GAT:
    return GAT(in_channels, hidden_channels, out_channels, num_heads, dropout, edge_dim)

def load_model_3layerGat(in_channels: int, hidden_channels: int, out_channels: int,
                         model_path: str = None, num_heads: int = 5, 
                         dropout: float = 0.3, edge_dim: int = None) -> GAT:
    model_path = model_path or 'app/mlruns/10/ec80f1fe720946fe9f11126ec5831338/artifacts/model/data/model.pth'
    model = get_raw_model(in_channels, hidden_channels, out_channels, num_heads, dropout, edge_dim)
    try:
        model = torch.load(model_path, map_location='cpu')
        model.eval()
    except Exception as e:
        raise RuntimeError(f"Error loading model weights: {e}")
    return model

def get_raw_cognn_model(gumbel_args, env_args, action_args) -> CoGNN:
    return CoGNN(gumbel_args, env_args, action_args)

def load_cognn_model(config, model_path: str) -> CoGNN:
    gumbel_args, env_args, action_args = config.Gumbel, config.Environment, config.Action
    model = get_raw_cognn_model(gumbel_args, env_args, action_args)
    try:
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
    except Exception as e:
        raise RuntimeError(f"Error loading model weights: {e}")
    return model
