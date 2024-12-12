from pyvis.network import Network

from pyvis.network import Network

def visualize_analysis_with_layers(subgraph_data, analysis_results, target, predicted_class, target_label):
    """
    Visualize the subgraph with layered attributes using Pyvis, maintaining consistent node positions.
    
    Args:
        subgraph_data (dict): Subgraph data containing node features, edge index, and target node index.
        analysis_results (dict): Layered analysis results with node contributions and edge weights.
        target (int): Index of the target node.
        predicted_class (int): Predicted class for the target node.
        target_label (int): Original label of the target node.
    """

    # Add all nodes to reference network to calculate positions
    nodes = set(node for edge in subgraph_data['edge_index'] for node in edge)
    
    
    def create_network(layer_name, graph_data, target, predicted_class):
        net = Network(height="800px", width="100%", bgcolor="#222222", font_color="white")
        
        # Disable physics to maintain positions
        net.set_options('''
        var options = {
          "physics": {
            "enabled": false
          }
        }
        ''')

        # Define colors for each label class
        label_colors = {
            0: "blue",
            1: "green",
            2: "purple",
            3: "orange",
            4: "cyan",
            5: "pink",
            6: "yellow"
        }

        # Add nodes with consistent positions
        for node in nodes:
            label = subgraph_data['labels'][node]
            contribution = next((n['normalized_contribution'] for n in graph_data['top_nodes'] if n['index'] == node), 0)
            size = 20 if node == target else 10 + contribution * 50
            color = "black" if node == target else label_colors.get(label, "white")
            
            if contribution > 0.01 or node == target:
                node_label = (
                    f"Target Node (Predicted: {predicted_class}, Original: {target_label})"
                    if node == target
                    else f"Node {node}, Label: {label}\nContribution: {contribution:.4f}"
                )
            else:
                node_label = str(label)

            # Use the same x, y coordinates as in reference network
            net.add_node(
                node,
                label=node_label,
                size=size,
                color=color,
                title=node_label,
            )

        # Add edges with weights
        for edge in subgraph_data['edge_index']:
            source, target_edge = edge
            weight = next((e['normalized_weight'] for e in graph_data['top_edges'] 
                         if e['source'] == source and e['target'] == target_edge), 0)
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

    print("Graphs saved as HTML files for each layer and overall, with consistent node positions.")



def visualize_subgraph_with_layers_weightened(
    edge_index, edge_weights_by_layer, node_labels, target_node_idx, predicted_class, target_label, save=True
):
    """
    Visualize subgraphs for each layer based on edge contributions using Pyvis with explicit directed edges,
    count edges with weight = 1, and maintain robust node positioning.

    Args:
        edge_index (list): Edge indices of shape [2, num_edges], where the first row is source nodes and the second row is target nodes.
        edge_weights_by_layer (list): List of edge weights for each layer. Each element is a list of weights (1 or 0).
        node_labels (list): Labels for each node.
        target_node_idx (int): Index of the target node.
        predicted_class (int): Predicted class for the target node.
        target_label (int): Original label of the target node.
        save (bool): Whether to save visualizations as HTML files.
    """
    # Get all unique nodes
    unique_nodes = list(set(edge_index[0] + edge_index[1]))  # Combine sources and targets

    def create_network_and_count_edges(layer_idx, edge_index, edge_weights, target_node_idx, predicted_class, target_label):
        from pyvis.network import Network
        import math

        # Create network with directed graph option
        net = Network(height="800px", width="100%", bgcolor="#222222", font_color="white", directed=True)

        # Disable physics to maintain positions
        net.set_options('''
        var options = {
          "physics": {
            "enabled": false
          }
        }
        ''')

        # Define colors for each label class
        label_colors = {
            0: "blue",
            1: "green",
            2: "purple",
            3: "orange",
            4: "cyan",
            5: "pink",
            6: "yellow"
        }

        # Calculate circular layout manually
        def circular_layout(nodes):
            layout = {}
            n = len(nodes)
            radius = 500  # Adjust radius as needed
            for i, node in enumerate(nodes):
                angle = 2 * math.pi * i / n
                layout[node] = {
                    'x': radius * math.cos(angle),
                    'y': radius * math.sin(angle)
                }
            return layout

        # Create manual layout
        node_positions = circular_layout(unique_nodes)

        # Add nodes with manual positions
        for node in unique_nodes:
            label = node_labels[node]
            color = "red" if node == target_node_idx else label_colors.get(label, "white")
            size = 20 if node == target_node_idx else 10
            node_title = (
                f"Target Node (Predicted: {predicted_class}, Original: {target_label})"
                if node == target_node_idx
                else f"Node {node}, Label: {label}"
            )

            net.add_node(
                node,
                label=f"Node {node}",
                size=size,
                color=color,
                title=node_title,
          #      x=node_positions[node]['x'],
           #     y=node_positions[node]['y']
            )

        # Count edges with weight = 1
        kept_edges_count = 0

        # Add directed edges based on weights
        for i, (source, target) in enumerate(zip(edge_index[0], edge_index[1])):
            if edge_weights[i] == 1.0:  # Only add edges with weight 1
                kept_edges_count += 1
                net.add_edge(
                    source, target,
                    title=f"Edge: {source} -> {target} (Contributes)",
                    arrows="to"  # Explicitly set arrow direction
                )

        return net, kept_edges_count

    # Visualize each layer and count edges
    total_kept_edges = 0
    for layer_idx, edge_weights in enumerate(edge_weights_by_layer):
        net, kept_edges_count = create_network_and_count_edges(
            layer_idx, edge_index, edge_weights, target_node_idx, predicted_class, target_label
        )
        print(f"Kept edges for layer {layer_idx}: {kept_edges_count}")
        total_kept_edges += kept_edges_count
        if save:
            filename = f'weightened_subgraph_layer_{layer_idx}.html'
            net.write_html(filename)
            print(f"Layer {layer_idx} visualization saved to {filename}")


def visualize_subgraph_pyvis(subgraph_data, save=True):
    """
    Visualize the extracted subgraph using Pyvis with consistent node positions.

    Args:
        subgraph_data (dict): Dictionary containing subgraph data with:
            - 'node_features': List of node features
            - 'edge_index': List of edges
            - 'target_node_idx': Index of the target node in the subgraph
            - 'labels': List of node labels
    """
    # Create reference network for position calculation
    reference_net = Network(height="800px", width="100%", bgcolor="#222222", font_color="white")
    
    # Set physics options for initial position calculation
    reference_net.set_options('''
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
    
    # Get all unique nodes for reference network
    nodes = set(node for edge in edge_index for node in edge)
    
    # Add nodes to reference network
    for node in nodes:
        reference_net.add_node(node)
    
    # Add edges to reference network
    for edge in edge_index:
        source, target = edge
        reference_net.add_edge(source, target)
    
    # Save reference layout
    reference_net.write_html('reference_layout.html')
    
    # Create main visualization network
    net = Network(height="800px", width="100%", bgcolor="#222222", font_color="white")
    
    # Disable physics to maintain positions
    net.set_options('''
    var options = {
      "physics": {
        "enabled": false
      }
    }
    ''')
    
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
    
    # Function to map node sizes
    def get_node_size(idx):
        return 20 if idx == target_node_idx else 10
    
    def get_node_color(idx):
        # Target node is always black; others depend on their labels
        return "black" if idx == target_node_idx else label_colors.get(labels[idx], "white")
    
    # Add nodes with consistent positions
    for idx in nodes:
        label_info = f"Label: {labels[idx]}"
        node_title = f"Target Node, {label_info}" if idx == target_node_idx else f"Node {idx}, {label_info}"
        net.add_node(
            idx, 
            label=f"Node {idx}" if idx != target_node_idx else f"{idx} Target Node with Label ({labels[idx]})",
            size=get_node_size(idx), 
            color=get_node_color(idx), 
            title=node_title,
#            x=reference_net.nodes[idx]['x'],
 #           y=reference_net.nodes[idx]['y']
        )
    
    # Add edges
    for edge in edge_index:
        source, target = edge
        net.add_edge(source, target, title=f"Edge {source} -> {target}")
    
    # Save and show the network
    if save:
        net.write_html('subgraph_visualization.html')
    return net.html

def visualize_subgraph_pyvis_no_labels(subgraph_data, save=True):
    """
    Visualize the extracted subgraph using Pyvis with consistent node positions, without labels.

    Args:
        subgraph_data (dict): Dictionary containing subgraph data with:
            - 'node_features': List of node features
            - 'edge_index': List of edges
            - 'target_node_idx': Index of the target node in the subgraph
    """
    # Create reference network for position calculation
    reference_net = Network(height="800px", width="100%", bgcolor="#222222", font_color="white")
    
    # Set physics options for initial position calculation
    reference_net.set_options('''
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
    
    # Get all unique nodes for reference network
    nodes = set(node for edge in edge_index for node in edge)
    
    # Add nodes to reference network
    for node in nodes:
        reference_net.add_node(node)
    
    # Add edges to reference network
    for edge in edge_index:
        source, target = edge
        reference_net.add_edge(source, target)
    
    # Save reference layout
    reference_net.write_html('reference_layout.html')
    
    # Create main visualization network
    net = Network(height="800px", width="100%", bgcolor="#222222", font_color="white")
    
    # Disable physics to maintain positions
    net.set_options('''
    var options = {
      "physics": {
        "enabled": false
      }
    }
    ''')
    
    # Function to map node sizes and colors
    def get_node_size(idx):
        return 15 if idx == target_node_idx else 10
    
    def get_node_color(idx):
        return 'red' if idx == target_node_idx else 'blue'
    
    # Add nodes with consistent positions
    for idx in nodes:
        net.add_node(
            idx, 
            label=f"Node {idx}", 
            size=get_node_size(idx), 
            color=get_node_color(idx), 
            title=f"Node {idx}",
            x=reference_net.nodes[idx]['x'],
            y=reference_net.nodes[idx]['y']
        )
    
    # Add edges
    for edge in edge_index:
        source, target = edge
        net.add_edge(source, target, title=f"Edge {source} -> {target}")
    
    # Save and show the network
    if save:
        net.write_html('subgraph_visualization.html')
    return net.html

def visualize_analysis_with_layers_no_labels(subgraph_data, analysis_results, target, predicted_class=1):
    """
    Visualize the subgraph with layered attributes using Pyvis, maintaining consistent node positions.
    No label version.
    
    Args:
        subgraph_data (dict): Subgraph data containing node features and edge index.
        analysis_results (dict): Layered analysis results with node contributions and edge weights.
        target (int): Index of the target node.
        predicted_class (int): Predicted class for the target node.
    """
    # Create reference network for position calculation
    reference_net = Network(height="800px", width="100%", bgcolor="#222222", font_color="white")
    
    reference_net.set_options('''
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

    # Add all nodes and edges to reference network
    nodes = set(node for edge in subgraph_data['edge_index'] for node in edge)
    for node in nodes:
        reference_net.add_node(node)
    
    for edge in subgraph_data['edge_index']:
        reference_net.add_edge(edge[0], edge[1])
    
    # Save reference layout
    reference_net.write_html('reference_layout.html')
    
    def create_network(layer_name, graph_data, target, predicted_class):
        net = Network(height="800px", width="100%", bgcolor="#222222", font_color="white")
        
        # Disable physics to maintain positions
        net.set_options('''
        var options = {
          "physics": {
            "enabled": false
          }
        }
        ''')

        # Add nodes with consistent positions
        for node in nodes:
            contribution = next((n['normalized_contribution'] for n in graph_data['top_nodes'] if n['index'] == node), 0)
            size = 10 + contribution * 50
            color = 'red' if node == target else ('yellow' if contribution > 0.01 else 'blue')
            
            label = ('target' if node == target else 
                    f"Node {node}\nContribution: {contribution:.4f}" if contribution > 0 else None)

            net.add_node(
                node,
                label=label,
                size=size,
                color=color,
                title=f"Node {node}",
                #x=reference_net.nodes[node]['x'],
                #y=reference_net.nodes[node]['y']
            )

        # Add edges with weights
        for edge in subgraph_data['edge_index']:
            source, target_edge = edge
            weight = next((e['normalized_weight'] for e in graph_data['top_edges'] 
                         if e['source'] == source and e['target'] == target_edge), 0)
            
            label = f"Edge Weight: {weight:.4f}" if weight > 0 else None
            
            net.add_edge(
                source, target_edge,
                width=1 + weight * 10,
                label=label
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

    print("Graphs saved as HTML files for each layer and overall, with consistent node positions.")