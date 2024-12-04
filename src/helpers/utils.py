import torch
from torch_geometric.utils import k_hop_subgraph
import numpy as np
from torch_geometric.utils import add_self_loops

def softmax_normalize(values):
    """
    Normalize values using softmax to create a probability distribution.
    
    :param values: Array of numeric values
    :return: Normalized values summing to 1
    """
    exp_values = np.exp(values)
    return exp_values / np.sum(exp_values)


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


def make_bidirectional(edge_index):
    print(f"edge_index shape before bidirect: {edge_index.shape}")
    edge_index = edge_index.t()
    print(f"Transposed edge_index shape: {edge_index.shape}")
    edge_index_reversed = edge_index.flip(0)
    print(f"Reversed edge_index shape: {edge_index_reversed.shape}")
    edge_index_bidirectional = torch.cat([edge_index, edge_index_reversed], dim=0)  # this should be 1 originally
    print(f"Concatenated bidirectional edge_index shape: {edge_index_bidirectional.shape}")
    edge_index_bidirectional = torch.unique(edge_index_bidirectional, dim=1)
    print(f"Unique bidirectional edge_index shape: {edge_index_bidirectional.shape}")
    return edge_index_bidirectional.t() #for not being cora



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


def extract_subgraph(node_idx, num_hops, node_features, edges):
    """
    Extract a k-hop subgraph for the specified node index.

    Args:
        node_idx (int): Target node index.
        num_hops (int): Number of hops for the subgraph.
        node_features (torch.Tensor): Tensor of node features.
        edges (torch.Tensor): Tensor of edges (shape: [num_edges, 2]).

    Returns:
        dict: Subgraph data with node features and edge index.
    """
    # Transpose edges for PyTorch Geometric compatibility
    edge_index = edges

    print(f"Original graph after transposition: {edge_index.shape}")

    # Extract the k-hop subgraph
    subset_nodes, subgraph_edge_index, mapping, edge_mask  = k_hop_subgraph(
        node_idx=node_idx, num_hops=num_hops, edge_index=edge_index, relabel_nodes=True
    )

    print(f"SubGraph edge_index after k_hop: {subgraph_edge_index.shape}")

    # Subset node features
    subgraph_node_features = node_features[subset_nodes]
    
    # Print dataset statistics
    print(f"Node feature matrix shape of subgraph(x): {subgraph_node_features.shape}")
    #print(f"Label tensor shape (y): {subgraph_edge_index.shape}")
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
    
    print(f"Subset nodes in subgraph: {subset_nodes}")
    print(f"Target node original index: {node_idx}")
    print(f"Target node index in subgraph: {target_node_subgraph_idx}")


    # check if this fixes it, it actually does. There is some magic in transpositions, as they appear everywhere in the code... not good
    subgraph_edge_index = subgraph_edge_index.t()
    print(subgraph_edge_index)

    return {
        "node_features": subgraph_node_features.tolist(),
        "edge_index": subgraph_edge_index.tolist(),
        "target_node_idx": target_node_subgraph_idx
    }

