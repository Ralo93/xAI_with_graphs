from pyvis.network import Network

def visualize_subgraph_with_consistent_layout_re(
    edge_index, edge_weights_by_layer, node_labels, target_node_idx, predicted_class, target_label, save=True
):
    """
    Visualize subgraphs for each layer based on edge contributions using Pyvis with explicit directed edges,
    count edges with weight = 1, and ensure the same node positioning across all layers.

    Additionally, save the original graph with the same layout.

    Args:
        edge_index (list): Edge indices of shape [2, num_edges], where the first row is source nodes and the second row is target nodes.
        edge_weights_by_layer (list): List of edge weights for each layer. Each element is a list of weights (1 or 0).
        node_labels (list): Labels for each node.
        target_node_idx (int): Index of the target node.
        predicted_class (int): Predicted class for the target node.
        target_label (int): Original label of the target node.
        save (bool): Whether to save visualizations as HTML files.
    """
    from pyvis.network import Network
    import math

    # Get all unique nodes
    unique_nodes = list(set(edge_index[0] + edge_index[1]))

    # Define colors for each of the 18 classes
    label_colors = {
        0: "#1f77b4", 1: "#ff7f0e", 2: "#2ca02c", 3: "#d62728", 4: "#9467bd",
        5: "#8c564b", 6: "#e377c2", 7: "#7f7f7f", 8: "#bcbd22", 9: "#17becf",
        10: "#a52a2a", 11: "#5f9ea0", 12: "#ff69b4", 13: "#4682b4", 14: "#d2691e",
        15: "#9acd32", 16: "#dda0dd", 17: "#cd5c5c"
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

    # Create manual layout (consistent for all layers)
    node_positions = circular_layout(unique_nodes)

    def create_network(layer_idx, edge_index, edge_weights=None, directed=True):
        # Create network with directed or undirected graph option
        net = Network(height="800px", width="100%", bgcolor="#222222", font_color="white", directed=directed)

        # Disable physics to maintain consistent positions
        net.set_options('''
        var options = {
          "physics": {
            "enabled": false
          }
        }
        ''')

        # Add nodes with predefined positions
        for node in unique_nodes:
            label = node_labels[node]
            color = "red" if node == target_node_idx else label_colors.get(label, "white")
            size = 40 if node == target_node_idx else 10
            node_title = (
                f"Target Node (Predicted: {predicted_class}, Original: {target_label})"
                if node == target_node_idx
                else f"Node with class {label}"
            )

            net.add_node(
                node,
                label=f"Target Node {node}" if node == target_node_idx else f"{label}",
                size=size,
                color=color,
                title=node_title,
                x=node_positions[node]['x'],
                y=node_positions[node]['y']
            )

        # Count edges with weight = 1
        kept_edges_count = 0

        # Add edges
        for i, (source, target) in enumerate(zip(edge_index[0], edge_index[1])):
            if edge_weights is None or edge_weights[i] == 1.0:  # Add all edges or only edges with weight 1
                kept_edges_count += 1
                net.add_edge(
                    source, target,
                    title=f"Edge: {source} -> {target} (Contributes)" if edge_weights else f"Edge: {source} -> {target}"
                )
                if not directed:
                    net.add_edge(
                        target, source,
                        title=f"Edge: {target} -> {source} (Contributes)" if edge_weights else f"Edge: {target} -> {source}"
                    )

        return net, kept_edges_count

    # Save the original graph
    if save:
        original_net, _ = create_network(-1, edge_index, directed=False)
        original_filename = 'consistent_layout_original_graph.html'
        original_net.write_html(original_filename)
        print(f"Original graph visualization saved to {original_filename}")

    # Visualize each layer and count edges
    total_kept_edges = 0
    for layer_idx, edge_weights in enumerate(edge_weights_by_layer):
        net, kept_edges_count = create_network(
            layer_idx, edge_index, edge_weights
        )
        print(f"Kept edges for layer {layer_idx}: {kept_edges_count}")
        total_kept_edges += kept_edges_count
        if save:
            filename = f'consistent_layout_subgraph_layer_{layer_idx}.html'
            net.write_html(filename)
            print(f"Layer {layer_idx} visualization saved to {filename}")


def visualize_subgraph_with_consistent_layout(
    edge_index, edge_weights_by_layer, node_labels, target_node_idx, predicted_class, target_label, save=True
):
    """
    Visualize subgraphs for each layer based on edge contributions using Pyvis with explicit directed edges,
    count edges with weight = 1, and ensure the same node positioning across all layers.

    Additionally, save the original graph with the same layout.

    Args:
        edge_index (list): Edge indices of shape [2, num_edges], where the first row is source nodes and the second row is target nodes.
        edge_weights_by_layer (list): List of edge weights for each layer. Each element is a list of weights (1 or 0).
        node_labels (list): Labels for each node.
        target_node_idx (int): Index of the target node.
        predicted_class (int): Predicted class for the target node.
        target_label (int): Original label of the target node.
        save (bool): Whether to save visualizations as HTML files.
    """
    from pyvis.network import Network
    import math

    # Get all unique nodes
    unique_nodes = list(set(edge_index[0] + edge_index[1]))

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

    # Create manual layout (consistent for all layers)
    node_positions = circular_layout(unique_nodes)

    def create_network(layer_idx, edge_index, edge_weights=None, directed=True):
        # Create network with directed or undirected graph option
        net = Network(height="800px", width="100%", bgcolor="#222222", font_color="white", directed=directed)

        # Disable physics to maintain consistent positions
        net.set_options('''
        var options = {
          "physics": {
            "enabled": false
          }
        }
        ''')

        # Add nodes with predefined positions
        for node in unique_nodes:
            label = node_labels[node]
            color = "red" if node == target_node_idx else label_colors.get(label, "white")
            size = 40 if node == target_node_idx else 10
            node_title = (
                f"Target Node (Predicted: {predicted_class}, Original: {target_label})"
                if node == target_node_idx
                else f"Node with class {label}"
            )

            net.add_node(
                node,
                label=f"Target Node {node}" if node == target_node_idx else f"{label}",
                size=size,
                color=color,
                title=node_title,
                x=node_positions[node]['x'],
                y=node_positions[node]['y']
            )

        # Count edges with weight = 1
        kept_edges_count = 0

        # Add edges
        for i, (source, target) in enumerate(zip(edge_index[0], edge_index[1])):
            if edge_weights is None or edge_weights[i] == 1.0:  # Add all edges or only edges with weight 1
                kept_edges_count += 1
                net.add_edge(
                    source, target,
                    title=f"Edge: {source} -> {target} (Contributes)" if edge_weights else f"Edge: {source} -> {target}"
                )
                if not directed:
                    net.add_edge(
                        target, source,
                        title=f"Edge: {target} -> {source} (Contributes)" if edge_weights else f"Edge: {target} -> {source}"
                    )

        return net, kept_edges_count

    # Save the original graph
    if save:
        original_net, _ = create_network(-1, edge_index, directed=False)
        original_filename = 'consistent_layout_original_graph.html'
        original_net.write_html(original_filename)
        print(f"Original graph visualization saved to {original_filename}")

    # Visualize each layer and count edges
    total_kept_edges = 0
    for layer_idx, edge_weights in enumerate(edge_weights_by_layer):
        net, kept_edges_count = create_network(
            layer_idx, edge_index, edge_weights
        )
        print(f"Kept edges for layer {layer_idx}: {kept_edges_count}")
        total_kept_edges += kept_edges_count
        if save:
            filename = f'consistent_layout_subgraph_layer_{layer_idx}.html'
            net.write_html(filename)
            print(f"Layer {layer_idx} visualization saved to {filename}")

from pyvis.network import Network
import networkx as nx

def visualize_analysis_with_layers_and_importance(subgraph_data, analysis_results, target, predicted_class, target_label):
    """
    Visualize the subgraph with layered attributes using Pyvis, maintaining consistent node positions.

    Args:
        subgraph_data (dict): Subgraph data containing node features, edge index, and target node index.
        analysis_results (dict): Layered analysis results with node contributions and edge weights.
        target (int): Index of the target node.
        predicted_class (int): Predicted class for the target node.
        target_label (int): Original label of the target node.
    """

    # Validate inputs
    def validate_input_data(subgraph_data, analysis_results):
        if 'edge_index' not in subgraph_data or not isinstance(subgraph_data['edge_index'], list):
            raise ValueError("Invalid or missing 'edge_index' in subgraph_data.")
        if 'labels' not in subgraph_data or not isinstance(subgraph_data['labels'], dict):
            raise ValueError("Invalid or missing 'labels' in subgraph_data.")
        if 'layers' not in analysis_results or not isinstance(analysis_results['layers'], list):
            raise ValueError("Invalid or missing 'layers' in analysis_results.")
        if 'overall' not in analysis_results:
            raise ValueError("Missing 'overall' in analysis_results.")

    validate_input_data(subgraph_data, analysis_results)

    # Extract unique nodes
    nodes = set(node for edge in subgraph_data['edge_index'] for node in edge)

    # Create consistent node positions using NetworkX
    G = nx.Graph()
    G.add_edges_from(subgraph_data['edge_index'])
    positions = nx.spring_layout(G, seed=42)  # Seed for reproducibility

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
            label = subgraph_data['labels'][node]
            contribution = next((n['normalized_contribution'] for n in graph_data['top_nodes'] if n['index'] == node), 0)
            size = 20 if node == target else 10 + contribution * 50
            color = "red" if node == target else label_colors.get(label, "white")

            node_label = (
                f"Target Node (Predicted: {predicted_class}, Original: {target_label})"
                if node == target
                else f"Node {node}, Label: {label}\nContribution: {contribution:.4f}"
            )

            x, y = positions[node]
            net.add_node(
                node,
                label=node_label,
                size=size,
                color=color,
                title=node_label,
                x=x * 1000,
                y=y * 1000
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

    def create_sender_receiver_network(graph_data, is_sender):
        net = Network(height="800px", width="100%", bgcolor="#222222", font_color="white")

        # Disable physics to maintain positions
        net.set_options('''
        var options = {
          "physics": {
            "enabled": false
          }
        }
        ''')

        # Compute sending or receiving importance
        node_importance = {node: 0 for node in nodes}
        for edge in subgraph_data['edge_index']:
            source, target = edge
            for layer in analysis_results['layers']:
                for edge_info in layer['top_edges']:
                    if edge_info['source'] == source and edge_info['target'] == target:
                        if is_sender:
                            node_importance[source] += edge_info['normalized_weight']
                        else:
                            node_importance[target] += edge_info['normalized_weight']

        # Normalize importance
        max_importance = max(node_importance.values()) or 1
        for node in nodes:
            node_importance[node] /= max_importance

        # Add nodes with importance values
        for node in nodes:
            label = subgraph_data['labels'][node]
            size = 20 if node == target else 10 + node_importance[node] * 50
            color = "red" if node == target else label_colors.get(label, "white")

            x, y = positions[node]
            net.add_node(
                node,
                label=f"Node {node}\nImportance: {node_importance[node]:.2f}",
                size=size,
                color=color,
                title=f"Node {node}\nImportance: {node_importance[node]:.2f}",
                x=x * 1000,
                y=y * 1000
            )

        # Add edges
        for edge in subgraph_data['edge_index']:
            source, target = edge
            net.add_edge(source, target, width=1, color="grey")

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

    # Generate and save sender importance visualization
    sender_net = create_sender_receiver_network(subgraph_data, is_sender=True)
    sender_net.write_html('subgraph_sender_importance.html')

    # Generate and save receiver importance visualization
    receiver_net = create_sender_receiver_network(subgraph_data, is_sender=False)
    receiver_net.write_html('subgraph_receiver_importance.html')

    print("Graphs saved as HTML files for each layer, overall, sender importance, and receiver importance with consistent node positions.")


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
            color = "red" if node == target else label_colors.get(label, "white")
            
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
            size = 40 if node == target_node_idx else 10
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
                title=node_title
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
        return "red" if idx == target_node_idx else label_colors.get(labels[idx], "white")
    
    # Add nodes with consistent positions
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
        return 'black' if idx == target_node_idx else 'blue'
    
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
                title=f"Node {node}"
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



from collections import defaultdict, deque
from pyvis.network import Network


def compute_hop_distances(edge_index, target):
    """
    Compute hop distances for all nodes in the graph from the target node.
    
    Args:
        edge_index (list of tuple): List of edges in the graph (source, target).
        target (int): Target node index.
        
    Returns:
        dict: A dictionary mapping each node to its hop distance from the target.
    """
    hop_distances = {}
    graph = defaultdict(list)
    for source, target_edge in edge_index:
        graph[source].append(target_edge)
        graph[target_edge].append(source)  # Assuming undirected graph

    queue = deque([(target, 0)])
    visited = set()

    while queue:
        node, distance = queue.popleft()
        if node in visited:
            continue
        visited.add(node)
        hop_distances[node] = distance
        for neighbor in graph[node]:
            if neighbor not in visited:
                queue.append((neighbor, distance + 1))

    return hop_distances


def visualize_analysis_with_layers_correct(subgraph_data, analysis_results, target, predicted_class, target_label):
    """
    Visualize the subgraph with layered attributes using Pyvis, maintaining consistent node positions.
    
    Args:
        subgraph_data (dict): Subgraph data containing node features, edge index, and target node index.
        analysis_results (dict): Layered analysis results with node contributions and edge weights.
        target (int): Index of the target node.
        predicted_class (int): Predicted class for the target node.
        target_label (int): Original label of the target node.
    """
    edge_index = subgraph_data['edge_index']
    labels = subgraph_data['labels']
    
    # Compute hop distances for filtering
    hop_distances = compute_hop_distances(edge_index, target)

    def create_network(layer_name, graph_data, target, predicted_class, max_hop):
        net = Network(height="800px", width="100%", bgcolor="#222222", font_color="white")

        # Disable physics to maintain positions
        net.set_options('''
        var options = {
          "physics": {
            "enabled": false
          }
        }
        ''')

        label_colors = {
            0: "blue", 1: "green", 2: "purple", 3: "orange", 4: "cyan", 5: "pink", 6: "yellow"
        }

        # Add nodes with consistent positions, filtering by hop distance
        for node, distance in hop_distances.items():
            if distance > max_hop:
                continue
            label = labels[node]
            contribution = next((n['normalized_contribution'] for n in graph_data['top_nodes'] if n['index'] == node), 0)
            size = 20 if node == target else 10 + contribution * 50
            color = "red" if node == target else label_colors.get(label, "white")

            if contribution > 0.001 or node == target:
                node_label = (
                    f"Target Node (Predicted: {predicted_class}, Original: {target_label}) and contribution {contribution:.4f}"
                    if node == target
                    else f"Node {node}, Label: {label}\nContribution: {contribution:.4f}"
                )
            else:
                node_label = str(label)

            net.add_node(node, label=node_label, size=size, color=color, title=node_label)

        # Add edges, filtering by hop distance
        for edge in edge_index:
            source, target_edge = edge
            if hop_distances.get(source, float('inf')) > max_hop or hop_distances.get(target_edge, float('inf')) > max_hop:
                continue
            weight = next((e['normalized_weight'] for e in graph_data['top_edges'] 
                          if e['source'] == source and e['target'] == target_edge), 0)
            net.add_edge(source, target_edge, width=1 + weight * 10, title=f"Edge Weight: {weight:.4f}")

        return net

    # Generate and save visualizations for each layer
    max_hops = {1: 1, 2: 2, 3: 3}
    for layer in analysis_results['layers']:
        layer_number = layer['layer_number']
        net = create_network(f"Layer {layer_number}", layer, target, predicted_class, max_hops[layer_number])
        net.write_html(f'subgraph_layer_{layer_number}.html')

    # Generate and save overall visualization
    overall_graph = analysis_results['overall']
    net = create_network("Overall", overall_graph, target, predicted_class, 3)
    net.write_html('subgraph_overall.html')

    print("Graphs saved as HTML files for each layer and overall, with consistent node positions.")
