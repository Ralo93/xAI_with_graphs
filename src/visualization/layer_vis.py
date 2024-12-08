# src/visualization/layer_vis.py
from typing import Dict, Any, List, Optional, Set
from .base import BaseGraphVisualizer


class LayerVisualizer(BaseGraphVisualizer):
    """Visualizer for layer-wise attention analysis"""

    def __init__(self, use_labels: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.use_labels = use_labels
        self.label_colors = {
            0: "blue",
            1: "green",
            2: "purple",
            3: "orange",
            4: "cyan",
            5: "pink",
            6: "yellow",
        }

    def _get_node_attributes(
        self,
        node: int,
        graph_data: Dict[str, Any],
        subgraph_data: Dict[str, Any],
        target: int,
        predicted_class: int,
        target_label: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Get visual attributes for a node based on its properties and contributions

        Args:
            node: Node index
            graph_data: Layer-specific graph data
            subgraph_data: Overall subgraph data
            target: Target node index
            predicted_class: Predicted class for target node
            target_label: Original label of target node
        """
        # Get node contribution
        contribution = next(
            (
                n["normalized_contribution"]
                for n in graph_data["top_nodes"]
                if n["index"] == node
            ),
            0,
        )

        # Calculate node size based on contribution
        size = 20 if node == target else 10 + contribution * 50

        # Determine node color
        if self.use_labels:
            label = subgraph_data["labels"][node]
            color = "black" if node == target else self.label_colors.get(label, "white")
        else:
            color = (
                "red" if node == target else "yellow" if contribution > 0.01 else "blue"
            )

        # Create node label
        if contribution > 0.01 or node == target:
            if node == target and self.use_labels:
                label = f"Target Node (Predicted: {predicted_class}, Original: {target_label})"
            else:
                label = (
                    (
                        f"Node {node}, Label: {subgraph_data['labels'][node]}\n"
                        f"Contribution: {contribution:.4f}"
                    )
                    if self.use_labels
                    else f"Node {node}\nContribution: {contribution:.4f}"
                )
        else:
            label = None

        return {
            "label": label,
            "size": size,
            "color": color,
            "title": label or f"Node {node}",
        }

    def _get_edge_attributes(
        self, source: int, target: int, graph_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get visual attributes for an edge based on attention weights"""
        weight = next(
            (
                e["normalized_weight"]
                for e in graph_data["top_edges"]
                if e["source"] == source and e["target"] == target
            ),
            0,
        )

        return {"width": 1 + weight * 10, "title": f"Edge Weight: {weight:.4f}"}

    def _create_layer_visualization(
        self,
        layer_name: str,
        graph_data: Dict[str, Any],
        subgraph_data: Dict[str, Any],
        target: int,
        predicted_class: int,
        target_label: Optional[int] = None,
    ) -> Network:
        """Create visualization for a single layer"""
        net = self._create_network()

        # Add nodes
        nodes = set(node for edge in subgraph_data["edge_index"] for node in edge)
        for node in nodes:
            attributes = self._get_node_attributes(
                node, graph_data, subgraph_data, target, predicted_class, target_label
            )
            net.add_node(node, **attributes)

        # Add edges
        for edge in subgraph_data["edge_index"]:
            source, target_node = edge
            attributes = self._get_edge_attributes(source, target_node, graph_data)
            net.add_edge(source, target_node, **attributes)

        return net

    def visualize(self, data: Dict[str, Any], save_dir: str = ".") -> List[str]:
        """
        Create visualizations for each layer and overall graph

        Args:
            data: Dictionary containing:
                - subgraph_data: Basic graph structure
                - analysis_results: Layer-wise analysis results
                - target: Target node index
                - predicted_class: Predicted class
                - target_label: Original label of target node
            save_dir: Directory to save visualizations

        Returns:
            List of paths to saved visualization files
        """
        saved_files = []

        # Generate layer visualizations
        for layer in data["analysis_results"]["layers"]:
            layer_number = layer["layer_number"]
            net = self._create_layer_visualization(
                f"Layer {layer_number}",
                layer,
                data["subgraph_data"],
                data["target"],
                data["predicted_class"],
                data.get("target_label"),
            )

            file_path = f"{save_dir}/subgraph_layer_{layer_number}.html"
            net.write_html(file_path)
            saved_files.append(file_path)

        # Generate overall visualization
        overall_net = self._create_layer_visualization(
            "Overall",
            data["analysis_results"]["overall"],
            data["subgraph_data"],
            data["target"],
            data["predicted_class"],
            data.get("target_label"),
        )

        file_path = f"{save_dir}/subgraph_overall.html"
        overall_net.write_html(file_path)
        saved_files.append(file_path)

        return saved_files
