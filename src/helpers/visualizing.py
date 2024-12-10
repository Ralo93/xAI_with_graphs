from pyvis.network import Network

def visualize_analysis_with_layers(subgraph_data, analysis_results, target, predicted_class, target_label):
    """
    Visualize the subgraph with layered attributes using Pyvis.

    Args:
        subgraph_data (dict): Subgraph data containing node features, edge index, and target node index.
        analysis_results (dict): Layered analysis results with node contributions and edge weights.
        target (int): Index of the target node.
        predicted_class (int): Predicted class for the target node.
    """
    def create_network(layer_name, graph_data, target, predicted_class):
        # Create Pyvis network
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

        # Define colors for each label class (0 to 6)
        label_colors = {
            0: "blue",
            1: "green",
            2: "purple",
            3: "orange",
            4: "cyan",
            5: "pink",
            6: "yellow"
        }

        # Add nodes
        nodes = set(node for edge in subgraph_data['edge_index'] for node in edge)
        for node in nodes:
            label = subgraph_data['labels'][node]
            contribution = next((n['normalized_contribution'] for n in graph_data['top_nodes'] if n['index'] == node), 0)
            size = 20 if node == target else 10 + contribution * 50
            color = "black" if node == target else label_colors.get(label, "white")
            # Modify the target node label in the loop where nodes are added
            if contribution > 0.01 or node == target:
                node_label = (
                    f"Target Node (Predicted: {predicted_class}, Original: {target_label})"
                    if node == target
                    else f"Node {node}, Label: {label}\nContribution: {contribution:.4f}"
                )
            else:
                node_label = None  # No label displayed for nodes with low contribution


            net.add_node(
                node, 
                label=node_label, 
                size=size, 
                color=color, 
                title=node_label
            )

        # Add edges
        for edge in subgraph_data['edge_index']:
            source, target_edge = edge
            weight = next((e['normalized_weight'] for e in graph_data['top_edges'] if e['source'] == source and e['target'] == target_edge), 0)
            net.add_edge(
                source, target_edge, 
                width=1 + weight * 10, 
                title=f"Edge Weight: {weight:.4f}"
            )

        return net

    # Generate and save visualizations for each layer
    for layer in analysis_results['layers']:
        layer_number = layer['layer_number']
        net = create_network(f"Layer {layer_number}", layer, target, predicted_class)
        net.write_html(f'subgraph_layer_{layer_number}.html')

    # Generate and save overall visualization
    overall_graph = analysis_results['overall']
    net = create_network("Overall", overall_graph, target, predicted_class)
    net.write_html('subgraph_overall.html')

    print("Graphs saved as HTML files for each layer and overall.")



def visualize_analysis_with_layers_no_labels(subgraph_data, analysis_results, target, predicted_class=1):
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
            color = 'red' if node == target else ('yellow' if node in [n['index'] for n in graph_data['top_nodes']] and contribution > 0.01 else 'blue')

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




def visualize_subgraph_pyvis(subgraph_data, save=True):
    """
    Visualize the extracted subgraph using Pyvis.

    Args:
        subgraph_data (dict): Dictionary containing subgraph data with:
            - 'node_features': List of node features
            - 'edge_index': List of edges
            - 'target_node_idx': Index of the target node in the subgraph
            - 'labels': List of node labels
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
    edge_index = subgraph_data['edge_index']
    target_node_idx = subgraph_data['target_node_idx']
    labels = subgraph_data['labels']
    
    # Define colors for each label class (0 to 6)
    label_colors = {
        0: "blue",
        1: "green",
        2: "yellow",
        3: "orange",
        4: "purple",
        5: "pink",
        6: "cyan"
    }
    
    # Function to map node sizes and colors
    def get_node_size(idx):
        return 20 if idx == target_node_idx else 10
    
    def get_node_color(idx):
        # Target node is always black; others depend on their labels
        return "black" if idx == target_node_idx else label_colors.get(labels[idx], "white")
    
    # Add nodes
    nodes = set(node for edge in edge_index for node in edge)  # Get all unique nodes
    for idx in nodes:
        label_info = f"Label: {labels[idx]}"
        node_title = f"Target Node, {label_info}" if idx == target_node_idx else f"Node {idx}, {label_info}"
        net.add_node(
            idx, 
            label=f"Node {idx}" if idx != target_node_idx else f"{idx} Target Node with Label ({labels[idx]})",
            size=get_node_size(idx), 
            color=get_node_color(idx), 
            title=node_title
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



def visualize_subgraph_pyvis_no_labels(subgraph_data, save=True):
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



from pyvis.network import Network
def visualize_subgraph_with_layers_weightened(edge_index, edge_weights_by_layer, node_labels, target_node_idx, predicted_class, target_label, save=True):
    """
    Visualize subgraphs for each layer based on edge contributions using Pyvis with explicit directed edges.

    Args:
        edge_index (list): Edge indices of shape [2, num_edges], where the first row is source nodes and the second row is target nodes.
        edge_weights_by_layer (list): List of edge weights for each layer. Each element is a list of weights (1 or 0).
        node_labels (list): Labels for each node.
        target_node_idx (int): Index of the target node.
        predicted_class (int): Predicted class for the target node.
        target_label (int): Original label of the target node.
        save (bool): Whether to save visualizations as HTML files.
    """
    def create_network(layer_idx, edge_index, edge_weights, target_node_idx, predicted_class, target_label):
        # Create a Pyvis network with directed graph option
        net = Network(height="800px", width="100%", bgcolor="#222222", font_color="white", directed=True)

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

        # Define colors for each label class (0 to 6)
        label_colors = {
            0: "blue",
            1: "green", 
            2: "purple",
            3: "orange",
            4: "cyan",
            5: "pink",
            6: "yellow"
        }

        # Add nodes
        unique_nodes = set(edge_index[0] + edge_index[1])  # Combine sources and targets
        for node in unique_nodes:
            label = node_labels[node]
            color = "red" if node == target_node_idx else label_colors.get(label, "white")
            size = 20 if node == target_node_idx else 10
            node_title = (
                f"Target Node (Predicted: {predicted_class}, Original: {target_label})"
                if node == target_node_idx
                else f"Node {node}, Label: {label}"
            )
            net.add_node(node, label=f"Node {node}", size=size, color=color, title=node_title)

        # Add directed edges based on weights
        for i, (source, target) in enumerate(zip(edge_index[0], edge_index[1])):
            if edge_weights[i] == 1.0:  # Only add edges with weight 1
                net.add_edge(
                    source, target, 
                    title=f"Edge: {source} -> {target} (Contributes)",
                    arrows="to"  # Explicitly set arrow direction
                )

        return net

    # Visualize each layer
    for layer_idx, edge_weights in enumerate(edge_weights_by_layer):
        net = create_network(layer_idx, edge_index, edge_weights, target_node_idx, predicted_class, target_label)
        if save:
            filename = f'weightend_subgraph_layer_{layer_idx}.html'
            net.write_html(filename)
            print(f"Layer {layer_idx} visualization saved to {filename}")

    print("All layer graphs saved.")

