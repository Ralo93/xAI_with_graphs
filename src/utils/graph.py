from typing import Tuple, List, Dict, Any
import torch
import numpy as np
from torch_geometric.utils import k_hop_subgraph, to_undirected, add_self_loops


class GraphUtils:
    """Utility functions for graph operations"""

    @staticmethod
    def make_bidirectional(edge_index: torch.Tensor) -> torch.Tensor:
        """Convert directed edges to undirected by adding reverse edges"""
        return to_undirected(edge_index)

    @staticmethod
    def create_sample_graph(
        num_nodes: int = 50, num_features: int = 1433, seed: int = 42
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create a sample graph for testing"""
        torch.manual_seed(seed)

        node_features = torch.rand((num_nodes, num_features), dtype=torch.float32)
        num_edges = num_nodes * 3
        row = torch.randint(0, num_nodes, (num_edges,), dtype=torch.long)
        col = torch.randint(0, num_nodes, (num_edges,), dtype=torch.long)
        edges = torch.stack([row, col], dim=0)

        return node_features, edges

    @staticmethod
    def extract_subgraph(
        node_idx: int,
        num_hops: int,
        node_features: torch.Tensor,
        edges: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, List]:
        """Extract a k-hop subgraph around a target node"""
        edge_index = edges

        # Extract k-hop subgraph
        subset_nodes, subgraph_edge_index, mapping, _ = k_hop_subgraph(
            node_idx=node_idx,
            num_hops=num_hops,
            edge_index=edge_index,
            relabel_nodes=True,
        )

        # Process subgraph data
        subgraph_node_features = node_features[subset_nodes]
        subgraph_edge_index = GraphUtils.make_bidirectional(subgraph_edge_index)

        # Ensure correct edge index format
        if subgraph_edge_index.dim() != 2 or subgraph_edge_index.size(0) != 2:
            subgraph_edge_index = subgraph_edge_index.t()

        # Add self-loops
        subgraph_edge_index = add_self_loops(edge_index=subgraph_edge_index)[0].t()

        # Get target node mapping
        target_node_subgraph_idx = mapping[0].item()

        return {
            "node_features": subgraph_node_features.tolist(),
            "edge_index": subgraph_edge_index.tolist(),
            "target_node_idx": target_node_subgraph_idx,
            "labels": labels[subset_nodes].tolist(),
        }
