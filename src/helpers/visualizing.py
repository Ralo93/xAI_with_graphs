

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

            if node == target:
                label = 'target'
            else:
                label = f"Node {node}\nContribution: {contribution:.4f}" if contribution > 0 else None
    

            net.add_node(
                node, 
                label=label,
                size=size, 
                color=color,
                title=f"Node {node}"
            )
        
        # Add edges
        for edge in subgraph_data['edge_index']:
            source, target = edge
            weight = next((e['normalized_weight'] for e in graph_data['top_edges'] if e['source'] == source and e['target'] == target), 0)

            label = f"Edge Weight: {weight:.4f}" if weight > 0 else None
    
            net.add_edge(
                source, target, 
                width=1 + weight * 10, 
                label=label
            )
        
        # Highlight predicted class
        #net.add_node(
        #    target, 
        #    label=f"Target Node {target} (Predicted Class: {predicted_class})",
        #    size=20, 
        #    color='green',
        #    title=f"Target Node {target}\nPredicted Class: {predicted_class}"
        #)
        
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

