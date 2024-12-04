import requests
import torch
from torch_geometric.utils import k_hop_subgraph
import matplotlib as plt
import numpy as np
from pyvis.network import Network
import json

# URL of your local FastAPI endpoint
URL = "http://localhost:8000/predict/"

CLOUD_URL = "https://cora-gat-image-196616273613.europe-west10.run.app/predict/"

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

    net = visualize_subgraph_pyvis(input_data_dict, save=False)


    # The subgraph extraction function now also has the target node we want to run prediction for
    target_node_idx = input_data_dict['target_node_idx']

    # get it out again because the request does not expect it
    del input_data_dict['target_node_idx']


    #try:
    print("Sending Subgraph to Prediction Endpoint...")
    response = requests.post(URL, json=input_data_dict)

    # Check response status
    response.raise_for_status()

    # Parse the response
    result = response.json()
    
    print("Prediction Response Received.")

    # Extract and process attention weights
    aw = result.get('attention_weights', [])
    
    normalized_analysis = analyze_attention_weights(result)

    print(normalized_analysis)

    visualize_analysis_with_layers(input_data_dict, normalized_analysis, target_node_idx)

    return result, target_node_idx

    #except requests.exceptions.RequestException as e:
    #    print(f"Error connecting to the endpoint: {e}")
    #except Exception as e:
    #    print(f"An error occurred: {e}")


from pyvis.network import Network

def visualize_analysis_with_layers(subgraph_data, analysis_results, target, predicted_class=1):
    """
    Visualize the subgraph with layered attributes using Pyvis.
    
    Args:
        subgraph_data (dict): Subgraph data containing node features, edge index, and target node index.
        analysis_results (dict): Layered analysis results with node contributions and edge weights.
        predicted_class (int): Predicted class for the target node.
    """
    def create_network(layer_name, graph_data, target, predicted_class):
        # Initialize Pyvis network
        net = Network(height="800px", width="100%", 
                      bgcolor="#222222", 
                      font_color="white")
        
        # Set global network physics options
        net.set_options('''
        var options = {
          "physics": {
            "forceAtlas2Based": {
              "gravitationalConstant": -50,
              "centralGravity": 0.01,
              "springLength": 100,
              "springConstant": 0.08
            },
            "minVelocity": 0.75,
            "solver": "forceAtlas2Based"
          }
        }
        ''')
        
        # Add nodes
        nodes = set(node for edge in subgraph_data['edge_index'] for node in edge)
        for node in nodes:
            contribution = next((n['normalized_contribution'] for n in graph_data['top_nodes'] if n['index'] == node), 0)
            size = 10 + contribution * 50
            color = 'red' if node == target else ('yellow' if node in [n['index'] for n in graph_data['top_nodes']] else 'blue')
            net.add_node(
                node, 
                label=f"Node {node}",
                size=size, 
                color=color,
                title=f"Node {node}\nContribution: {contribution:.4f}"
            )
        
        # Add edges
        for edge in subgraph_data['edge_index']:
            source, target = edge
            weight = next((e['normalized_weight'] for e in graph_data['top_edges'] if e['source'] == source and e['target'] == target), 0)
            net.add_edge(
                source, target, 
                width=1 + weight * 10, 
                title=f"Edge Weight: {weight:.4f}"
            )
        
        # Highlight predicted class
        net.add_node(
            target, 
            label=f"Target Node {target} (Predicted Class: {predicted_class})",
            size=20, 
            color='green',
            title=f"Target Node {target}\nPredicted Class: {predicted_class}"
        )
        
        return net

    # Iterate through layers and overall to create separate graphs
    for layer in analysis_results['layers']:
        layer_number = layer['layer_number']
        net = create_network(f"Layer {layer_number}", layer, target, predicted_class)
        net.write_html(f'subgraph_layer_{layer_number}.html')
    
    # Create overall graph
    overall_graph = analysis_results['overall']
    net = create_network("Overall", overall_graph, target, predicted_class)
    net.write_html('subgraph_overall.html')

    print("Graphs saved as HTML files for each layer and overall.")



from pyvis.network import Network

def visualize_subgraph_pyvis(subgraph_data, save=True):
    """
    Visualize the extracted subgraph using Pyvis.

    Args:
        subgraph_data (dict): Dictionary containing subgraph data with:
            - 'node_features': List of node features
            - 'edge_index': List of edges
            - 'target_node_idx': Index of the target node in the subgraph
    """
    # Create a Pyvis network
    net = Network(height="800px", width="100%", 
                  bgcolor="#222222", 
                  font_color="white")
    
    # Set global network physics options
    net.set_options('''
    var options = {
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -50,
          "centralGravity": 0.01,
          "springLength": 100,
          "springConstant": 0.08
        },
        "minVelocity": 0.75,
        "solver": "forceAtlas2Based"
      }
    }
    ''')
    
    # Extract subgraph details
    #node_features = subgraph_data['node_features']
    edge_index = subgraph_data['edge_index']
    target_node_idx = subgraph_data['target_node_idx']
    
    # Function to map node sizes and colors
    def get_node_size(idx):
        return 15 if idx == target_node_idx else 10
    
    def get_node_color(idx):
        return 'red' if idx == target_node_idx else 'blue'
    
    # Add nodes
    nodes = set(node for edge in edge_index for node in edge)  # Get all unique nodes
    for idx in nodes:
        net.add_node(
            idx, 
            label=f"Node {idx}", 
            size=get_node_size(idx), 
            color=get_node_color(idx), 
            title=f"Node {idx}"
        )
    
    # Add edges
    for edge in edge_index:
        source, target = edge
        net.add_edge(
            source, target, 
            title=f"Edge {source} -> {target}"
        )
    
    # Save and show the network
    if save:
        net.write_html('subgraph_visualization.html')
    return net.html


# Usage
# html = visualize_attention_network(analysis_results)

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

# Example usage
# result = your_model_result

# Run the test
if __name__ == "__main__":

    target_node = 11  # Specify the node to predict
    result, target_node_idx = test_predict_endpoint(target_node)
    target_class_probabilities = result['class_probabilities'][target_node_idx]
    predicted_class = target_class_probabilities.index(max(target_class_probabilities))
    print(f"Predicted class for target node: {predicted_class}")

