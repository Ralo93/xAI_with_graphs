import torch
from torch_geometric.utils import k_hop_subgraph
import numpy as np
from torch_geometric.utils import add_self_loops

from collections import defaultdict
import numpy as np
from scipy.special import softmax

def analyze_attention_weights_enhanced(result, node_positions=None):
    """
    Enhanced analysis of attention weights with additional aggregations for interpretability.
    
    :param result: Dictionary containing attention weights and edge index.
    :param node_positions: Optional dictionary mapping node IDs to positions for positional analysis.
    :return: Structured analysis results with additional aggregations.
    """
    attention_weights = result.get('attention_weights', [])
    if not attention_weights:
        raise ValueError("No attention weights returned from the server.")
    
    edge_index = np.array(result['edge_index'])
    num_nodes = np.max(edge_index) + 1
    total_sum = np.zeros(len(edge_index[0]))
    analysis_results = {
        'layers': [],
        'overall': {
            'top_edges': [],
            'top_nodes': [],
            'aggregations': {}
        }
    }
    
    # Additional aggregations
    edge_type_contributions = defaultdict(float)
    degree_contributions = np.zeros(num_nodes)
    position_contributions = defaultdict(float) if node_positions else None
    symmetry_scores = []
    
    for layer_num, layer_weights in enumerate(attention_weights, 1):
        layer_weights = np.array(layer_weights)
        layer_sum = layer_weights.sum(axis=1)
        total_sum += layer_sum
        
        # Node contributions
        node_contributions = np.zeros(num_nodes)
        for (src, dest), weight in zip(edge_index.T, layer_sum):
            node_contributions[src] += weight
            node_contributions[dest] += weight
            degree_contributions[src] += weight
            degree_contributions[dest] += weight
            if position_contributions is not None:
                position_contributions[node_positions[src]] += weight
                position_contributions[node_positions[dest]] += weight
        
        # Symmetry analysis
        for (src, dest), weight in zip(edge_index.T, layer_sum):
            reverse_indices = np.where((edge_index[0] == dest) & (edge_index[1] == src))[0]
            if len(reverse_indices) > 0:
                reverse_weight = layer_sum[reverse_indices[0]]
                symmetry_scores.append(abs(weight - reverse_weight))
        
        # Normalizations
        normalized_layer_sum = softmax(layer_sum)
        normalized_node_contributions = softmax(node_contributions)
        
        # Top edges and nodes for this layer
        top_edge_indices = layer_sum.argsort()[::-1][:5]
        top_edges = [
            {
                'index': idx, 
                'source': edge_index[0][idx], 
                'target': edge_index[1][idx],
                'normalized_weight': normalized_layer_sum[idx]
            } for idx in top_edge_indices
        ]
        top_node_indices = node_contributions.argsort()[::-1][:5]
        top_nodes = [
            {
                'index': node, 
                'normalized_contribution': normalized_node_contributions[node]
            } for node in top_node_indices
        ]
        
        # Store layer results
        analysis_results['layers'].append({
            'layer_number': layer_num,
            'top_edges': top_edges,
            'top_nodes': top_nodes
        })
    
    # Overall results
    normalized_total_sum = softmax(total_sum)
    overall_node_contributions = np.zeros(num_nodes)
    for (src, dest), weight in zip(edge_index.T, total_sum):
        overall_node_contributions[src] += weight
        overall_node_contributions[dest] += weight
    
    normalized_overall_node_contributions = softmax(overall_node_contributions)
    top_overall_edge_indices = total_sum.argsort()[::-1][:5]
    top_overall_edges = [
        {
            'index': idx, 
            'source': edge_index[0][idx], 
            'target': edge_index[1][idx],
            'normalized_weight': normalized_total_sum[idx]
        } for idx in top_overall_edge_indices
    ]
    top_overall_node_indices = overall_node_contributions.argsort()[::-1][:5]
    top_overall_nodes = [
        {
            'index': node, 
            'normalized_contribution': normalized_overall_node_contributions[node]
        } for node in top_overall_node_indices
    ]
    
    analysis_results['overall'] = {
        'top_edges': top_overall_edges,
        'top_nodes': top_overall_nodes,
        'aggregations': {
            'symmetry_scores': symmetry_scores,
            'degree_contributions': degree_contributions.tolist(),
            'position_contributions': position_contributions if position_contributions else None
        }
    }
    
    return analysis_results


def softmax_normalize(values):
    """
    Normalize values using softmax to create a probability distribution.
    
    Args:
        values (np.ndarray): Array of numeric values.
    Returns:
        np.ndarray: Normalized values summing to 1.
    """
    exp_values = np.exp(values - np.max(values))  # Subtract max for numerical stability
    return exp_values / np.sum(exp_values)


def analyze_attention_weights_but_edges(result):
    """
    Analyze attention weights using softmax normalization.

    Args:
        result (dict): Dictionary containing attention weights and edge index.
    
    Returns:
        dict: Structured analysis results with normalized edge weights and node contributions.
    """
    # Extract attention weights and edge index
    attention_weights = result.get('attention_weights', [])
    edge_index = np.array(result.get('edge_index', []))

    if not attention_weights or edge_index.size == 0:
        raise ValueError("Invalid result: Missing attention weights or edge index.")
    
    num_nodes = np.max(edge_index) + 1  # Determine the total number of nodes
    total_sum = np.zeros(len(edge_index[0]))  # Initialize total weights for edges

    # Prepare results structure
    analysis_results = {
        'layers': [],
        'overall': {
            'top_edges': [],
            'top_nodes': []
        }
    }

    for layer_num, layer_weights in enumerate(attention_weights, start=1):
        # Sum weights across attention heads
        layer_weights = np.array(layer_weights)
        layer_sum = layer_weights.sum(axis=1)

        # Update total edge weights
        total_sum += layer_sum

        # Compute node contributions
        node_contributions = np.zeros(num_nodes)
        for (src, dest), weight in zip(edge_index.T, layer_sum):
            node_contributions[src] += weight
            node_contributions[dest] += weight

        # Normalize edge and node contributions using softmax
        normalized_layer_sum = softmax_normalize(layer_sum)
        normalized_node_contributions = softmax_normalize(node_contributions)

        # Identify top edges and nodes for this layer
        top_edges = [
            {
                'index': idx,
                'source': edge_index[0][idx],
                'target': edge_index[1][idx],
                'normalized_weight': normalized_layer_sum[idx]
            }
            for idx in layer_sum.argsort()[::-1][:5]
        ]
        top_nodes = [
            {
                'index': node,
                'normalized_contribution': normalized_node_contributions[node]
            }
            for node in node_contributions.argsort()[::-1][:5]
        ]

        # Store results for the current layer
        analysis_results['layers'].append({
            'layer_number': layer_num,
            'top_edges': top_edges,
            'top_nodes': top_nodes
        })

    # Calculate overall normalized contributions
    normalized_total_sum = softmax_normalize(total_sum)
    overall_node_contributions = np.zeros(num_nodes)
    for (src, dest), weight in zip(edge_index.T, total_sum):
        overall_node_contributions[src] += weight
        overall_node_contributions[dest] += weight
    normalized_overall_node_contributions = softmax_normalize(overall_node_contributions)

    # Identify top overall edges and nodes
    top_overall_edges = [
        {
            'index': idx,
            'source': edge_index[0][idx],
            'target': edge_index[1][idx],
            'normalized_weight': normalized_total_sum[idx]
        }
        for idx in total_sum.argsort()[::-1][:5]
    ]
    top_overall_nodes = [
        {
            'index': node,
            'normalized_contribution': normalized_overall_node_contributions[node]
        }
        for node in overall_node_contributions.argsort()[::-1][:5]
    ]

    # Store overall results
    analysis_results['overall'] = {
        'top_edges': top_overall_edges,
        'top_nodes': top_overall_nodes
    }

    return analysis_results


def analyze_attention_weights(result):
    """
    Analyze attention weights using softmax normalization.
    
    :param result: Dictionary containing attention weights and edge index
    :return: Structured analysis results
    """
    # Extract attention weights
    attention_weights = result.get('attention_weights', [])
    
    if not attention_weights:
        raise ValueError("No attention weights returned from the server.")
    
    # Prepare edge index and node information
    edge_index = np.array(result['edge_index'])
    num_nodes = np.max(edge_index) + 1
    
    # Prepare results structure
    analysis_results = {
        'layers': [],
        'overall': {
            'top_edges': [],
            'top_nodes': []
        }
    }
    
    # Total sum across all layers
    total_sum = np.zeros(len(edge_index[0]))
    
    # Per-layer analysis
    for layer_num, layer_weights in enumerate(attention_weights, 1):
        # Sum attention weights across heads
        layer_weights = np.array(layer_weights)
        layer_sum = layer_weights.sum(axis=1)
        
        # Update total sum
        total_sum += layer_sum
        
        # Calculate node contributions
        node_contributions = np.zeros(num_nodes)
        for (src, dest), weight in zip(edge_index.T, layer_sum):
            node_contributions[src] += weight
            node_contributions[dest] += weight
        
        # Softmax normalization for edges
        normalized_layer_sum = softmax_normalize(layer_sum)
        
        # Softmax normalization for node contributions
        normalized_node_contributions = softmax_normalize(node_contributions)
        
        # Top edges for this layer
        top_edge_indices = layer_sum.argsort()[::-1][:5]
        top_edges = [
            {
                'index': idx, 
                'source': edge_index[0][idx], 
                'target': edge_index[1][idx],
                'normalized_weight': normalized_layer_sum[idx]
            } for idx in top_edge_indices
        ]
        
        # Top nodes for this layer
        top_node_indices = node_contributions.argsort()[::-1][:5]
        top_nodes = [
            {
                'index': node, 
                'normalized_contribution': normalized_node_contributions[node]
            } for node in top_node_indices
        ]
        
        # Store layer results
        analysis_results['layers'].append({
            'layer_number': layer_num,
            'top_edges': top_edges,
            'top_nodes': top_nodes
        })
    
    # Overall softmax normalization
    normalized_total_sum = softmax_normalize(total_sum)
    
    # Calculate overall node contributions
    overall_node_contributions = np.zeros(num_nodes)
    for (src, dest), weight in zip(edge_index.T, total_sum):
        overall_node_contributions[src] += weight
        overall_node_contributions[dest] += weight
    
    normalized_overall_node_contributions = softmax_normalize(overall_node_contributions)
    
    # Top overall edges
    top_overall_edge_indices = total_sum.argsort()[::-1][:5]
    top_overall_edges = [
        {
            'index': idx, 
            'source': edge_index[0][idx], 
            'target': edge_index[1][idx],
            'normalized_weight': normalized_total_sum[idx]
        } for idx in top_overall_edge_indices
    ]
    
    # Top overall nodes
    top_overall_node_indices = overall_node_contributions.argsort()[::-1][:5]
    top_overall_nodes = [
        {
            'index': node, 
            'normalized_contribution': normalized_overall_node_contributions[node]
        } for node in top_overall_node_indices
    ]
    
    # Store overall results
    analysis_results['overall'] = {
        'top_edges': top_overall_edges,
        'top_nodes': top_overall_nodes
    }
    
    return analysis_results


from torch_geometric.utils import to_undirected

def make_bidirectional(edge_index):
    edge_index = to_undirected(edge_index)
    return edge_index


def create_large_graph(num_nodes=50, num_features=1433):
    """
    Create a large sample graph with PyTorch tensors for a graph neural network model.

    Args:
        num_nodes (int): Number of nodes in the graph.
        num_features (int): Number of features per node.

    Returns:
        tuple: A tuple containing node_features and edges as PyTorch tensors.
    """

    torch.manual_seed(42)  # Set the random seed
    # Generate random node features (num_nodes x num_features)
    node_features = torch.rand((num_nodes, num_features), dtype=torch.float32)

    # Generate random edges
    num_edges = num_nodes * 3  # Example: 2 edges per node on average
    row = torch.randint(0, num_nodes, (num_edges,), dtype=torch.long)
    col = torch.randint(0, num_nodes, (num_edges,), dtype=torch.long)
    edges = torch.stack([row, col], dim=0)

    print("Full Graph Generated:")
    print(f"Node Features Shape: {node_features.shape}")
    print(f"Edges Shape: {edges.shape}")

    return node_features, edges

def extract_subgraph(node_idx, num_hops, node_features, edges, labels, take_all=False):
    """
    Extract a k-hop subgraph for the specified node index, or return the whole graph if take_all is True.

    Args:
        node_idx (int): Target node index.
        num_hops (int): Number of hops for the subgraph.
        node_features (torch.Tensor): Tensor of node features.
        edges (torch.Tensor): Tensor of edges (shape: [num_edges, 2]).
        labels (torch.Tensor): Tensor of node labels.
        take_all (bool): If True, return the entire graph.

    Returns:
        dict: Subgraph or whole graph data with node features and edge index.
    """

    if take_all:
        # Return the entire graph
        print("Returning the entire graph.")
        edge_index = make_bidirectional(edges)
        edge_index = add_self_loops(edge_index=edge_index)[0]
        edge_index = edge_index.t()

                # Print dataset statistics
        print(f"Node feature matrix shape of subgraph(x) but the whole graph: {node_features.shape}")
        print(f"Edge index shape (original): {edge_index.shape}")

        return {
            "node_features": node_features.tolist(),
            "edge_index": edge_index.tolist(),
            "target_node_idx": node_idx,
            "labels": labels.tolist(),
        }

    # Transpose edges for PyTorch Geometric compatibility
    edge_index = edges

    print(f"Original graph after transposition: {edge_index.shape}")

    # Extract the k-hop subgraph
    subset_nodes, subgraph_edge_index, mapping, edge_mask = k_hop_subgraph(
        node_idx=node_idx, num_hops=num_hops, edge_index=edge_index, relabel_nodes=True
    )

    print(f"SubGraph edge_index after k_hop: {subgraph_edge_index.shape}")

    # Subset node features
    subgraph_node_features = node_features[subset_nodes]

    # Print dataset statistics
    print(f"Node feature matrix shape of subgraph(x): {subgraph_node_features.shape}")
    print(f"Edge index shape (original): {edge_index.shape}")
    print(f"Number of edges in subgraph size: {subgraph_edge_index.size(1)}")

    # Make bidirectional and add self-loops
    subgraph_edge_index = make_bidirectional(subgraph_edge_index)
    print(f"Edge index shape subgraph (after make_bidirectional): {subgraph_edge_index.shape}")

    if subgraph_edge_index.dim() != 2 or subgraph_edge_index.size(0) != 2:
        print(f"Transposing edge index. Current shape: {subgraph_edge_index.shape}")
        subgraph_edge_index = subgraph_edge_index.t()

    subgraph_edge_index = add_self_loops(edge_index=subgraph_edge_index)[0]
    print(f"Edge index shape subgraph (after add_self_loops): {subgraph_edge_index.shape}")

    print("Extracted Subgraph:")
    print(f"Subgraph Node Features Shape: {subgraph_node_features.shape}")
    print(f"Subgraph Edge Index Shape: {subgraph_edge_index.shape}")

    # Retrieve the new index of the target node in the subgraph
    target_node_subgraph_idx = mapping[0].item()

    #print(f"Subset nodes in subgraph: {subset_nodes}")
    print(f"Target node original index: {node_idx}")
    print(f"Target node index in subgraph: {target_node_subgraph_idx}")

    subgraph_edge_index = subgraph_edge_index.t()
    print(f"Subgraph target label (mapped): {labels[subset_nodes][target_node_subgraph_idx]}")

    print(f"Original target node label: {labels[node_idx]}")
    print(f"Subgraph target node label: {labels[subset_nodes][target_node_subgraph_idx]}")

    return {
        "node_features": subgraph_node_features.tolist(),
        "edge_index": subgraph_edge_index.tolist(),
        "target_node_idx": target_node_subgraph_idx,
        "labels": labels[subset_nodes].tolist(),  # Add subset labels for inspection
    }



