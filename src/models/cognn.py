import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import MessagePassing
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class CoGNNConfig:
    """Configuration for CoGNN model"""

    class Action:
        ACTIVATION = F.relu
        DROPOUT = 0.2
        HIDDEN_DIM = 64
        NUM_LAYERS = 2
        INPUT_DIM = 128
        OUTPUT_DIM = 2
        AGG = "mean"

    class Environment:
        INPUT_DIM = 1433  # Cora dataset input dimension
        OUTPUT_DIM = 7  # Cora dataset number of classes
        NUM_LAYERS = 3
        DROPOUT = 0.2
        HIDDEN_DIM = 128
        LAYER_NORM = False
        SKIP_CONNECTION = True
        AGG = "sum"

    class Gumbel:
        TEMPERATURE = 0.5
        TAU = 0.01
        LEARN_TEMPERATURE = False


class EncoderLinear(nn.Linear):
    def forward(self, x: Tensor, pestat=None) -> Tensor:
        return super().forward(x)


class WeightedGNNConv(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int, aggr="add", bias=True):
        super().__init__(aggr=aggr)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lin = nn.Linear(2 * in_channels, out_channels, bias=bias)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        out = self.propagate(
            edge_index, x=x, edge_attr=edge_attr, edge_weight=edge_weight
        )
        out = self.lin(torch.cat((x, out), dim=-1))
        return out

    def message(
        self,
        x_j: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        msg = x_j if edge_attr is None else x_j * edge_attr
        if edge_weight is not None:
            msg = msg * edge_weight.view(-1, 1)
        return msg


class ActionNet(nn.Module):
    def __init__(self, config=CoGNNConfig.Action):
        super().__init__()
        self.num_layers = config.NUM_LAYERS
        self.dropout = nn.Dropout(config.DROPOUT)
        self.act = config.ACTIVATION

        self.net = nn.ModuleList(
            [
                WeightedGNNConv(
                    config.INPUT_DIM if i == 0 else config.HIDDEN_DIM,
                    (
                        config.HIDDEN_DIM
                        if i < config.NUM_LAYERS - 1
                        else config.OUTPUT_DIM
                    ),
                    aggr=config.AGG,
                )
                for i in range(config.NUM_LAYERS)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        env_edge_attr: Optional[torch.Tensor] = None,
        act_edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        edge_attrs = [env_edge_attr] + [act_edge_attr] * (self.num_layers - 1)

        for idx, (edge_attr, layer) in enumerate(zip(edge_attrs, self.net)):
            x = layer(x, edge_index, edge_attr)
            if idx < len(self.net) - 1:
                x = self.dropout(x)
                x = self.act(x)
        return x


class CoGNN(nn.Module):
    def __init__(self, config: CoGNNConfig = CoGNNConfig()):
        super().__init__()

        env_args = config.Environment
        action_args = config.Action
        gumbel_args = config.Gumbel

        # Environment network
        self.env_net = nn.ModuleList(
            [EncoderLinear(env_args.INPUT_DIM, env_args.HIDDEN_DIM)]
            + [
                WeightedGNNConv(env_args.HIDDEN_DIM, env_args.HIDDEN_DIM)
                for _ in range(env_args.NUM_LAYERS - 1)
            ]
            + [nn.Linear(env_args.HIDDEN_DIM, env_args.OUTPUT_DIM)]
        )

        self.hidden_layer_norm = (
            nn.LayerNorm(env_args.HIDDEN_DIM) if env_args.LAYER_NORM else nn.Identity()
        )
        self.dropout = nn.Dropout(p=env_args.DROPOUT)

        # Gumbel softmax parameters
        self.learn_temp = gumbel_args.LEARN_TEMPERATURE
        self.temp = gumbel_args.TEMPERATURE

        # Action networks
        self.in_act_net = ActionNet(action_args)
        self.out_act_net = ActionNet(action_args)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        pestat=None,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
        edge_ratio_node_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        x = self.env_net[0](x, pestat)
        x = self.dropout(x)

        for layer_idx in range(1, len(self.env_net) - 1):
            x = self.hidden_layer_norm(x)

            in_logits = self.in_act_net(x, edge_index)
            out_logits = self.out_act_net(x, edge_index)

            in_probs = F.gumbel_softmax(logits=in_logits, tau=self.temp, hard=True)
            out_probs = F.gumbel_softmax(logits=out_logits, tau=self.temp, hard=True)

            edge_weight = self.create_edge_weight(
                edge_index, in_probs[:, 0], out_probs[:, 0]
            )
            x = self.env_net[layer_idx](x, edge_index, edge_weight=edge_weight)
            x = self.dropout(x)

        x = self.env_net[-1](x)
        edge_ratio_tensor = -1 * torch.ones(
            size=(len(self.env_net) - 2,), device=x.device
        )

        return x, edge_ratio_tensor

    def create_edge_weight(
        self,
        edge_index: torch.Tensor,
        keep_in_prob: torch.Tensor,
        keep_out_prob: torch.Tensor,
    ) -> torch.Tensor:
        u, v = edge_index
        edge_in_prob = keep_in_prob[v]
        edge_out_prob = keep_out_prob[u]
        return edge_in_prob * edge_out_prob
