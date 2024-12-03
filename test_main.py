import requests
import torch
from torch_geometric.utils import k_hop_subgraph

# URL of your local FastAPI endpoint
URL = "http://localhost:8000/predict/"


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


from torch_geometric.utils import add_self_loops

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
    return {
        "node_features": subgraph_node_features.tolist(),
        "edge_index": subgraph_edge_index.tolist(),
        "target_node_idx": target_node_subgraph_idx
    }

def test_predict_endpoint(target_node, num_hops=3):
    # Create a larger graph
    node_features, edges = create_large_graph()

    # Extract the subgraph for the target node
    input_data_dict = extract_subgraph(
        node_idx=target_node, num_hops=num_hops, node_features=node_features, edges=edges
    )

    target_node_idx = input_data_dict['target_node_idx']

    # get it out again because the request does not expect it
    del input_data_dict['target_node_idx']

    try:
        # Send POST request to the prediction endpoint
        print("Sending Subgraph to Prediction Endpoint...")
        response = requests.post(URL, json=input_data_dict)

        # Check response status
        response.raise_for_status()

        # Parse the response
        result = response.json()

        print("Prediction Response:")
        #print(f"Class Probabilities: {result['class_probabilities']}")

        return result, target_node_idx

    except requests.exceptions.RequestException as e:
        print(f"Error connecting to the endpoint: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")


# Run the test
if __name__ == "__main__":
    target_node = 10  # Specify the node to predict
    result, target_node_idx = test_predict_endpoint(target_node)
    target_class_probabilities = result['class_probabilities'][target_node_idx]
    predicted_class = target_class_probabilities.index(max(target_class_probabilities))
    print(f"Predicted class for target node: {predicted_class}")

