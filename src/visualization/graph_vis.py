from typing import Dict, Any, Optional, Set
from .base import BaseGraphVisualizer


class SubgraphVisualizer(BaseGraphVisualizer):
    """Visualizer for subgraphs with optional labels"""

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

    def _get_nodes(self, edge_index: list) -> Set[int]:
        """Get unique nodes from edge index"""
        return set(node for edge in edge_index for node in edge)

    def _get_node_attributes(
        self, node_idx: int, target_idx: int, labels: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Get visual attributes for a node"""
        is_target = node_idx == target_idx

        attributes = {
            "size": 20 if is_target else 10,
            "color": (
                "black"
                if is_target
                else (
                    self.label_colors.get(labels[node_idx], "white")
                    if labels and self.use_labels
                    else "blue"
                )
            ),
        }

        if self.use_labels and labels:
            label_info = f"Label: {labels[node_idx]}"
            attributes["label"] = (
                f"Target Node with Label ({labels[node_idx]})"
                if is_target
                else f"Node {node_idx}"
            )
            attributes["title"] = (
                f"Target Node, {label_info}"
                if is_target
                else f"Node {node_idx}, {label_info}"
            )
        else:
            attributes["label"] = "target" if is_target else f"Node {node_idx}"
            attributes["title"] = f"Node {node_idx}"

        return attributes

    def visualize(
        self, data: Dict[str, Any], save_path: Optional[str] = None
    ) -> Optional[str]:
        """Create subgraph visualization"""
        net = self._create_network()

        # Add nodes
        nodes = self._get_nodes(data["edge_index"])
        for node in nodes:
            attributes = self._get_node_attributes(
                node, data["target_node_idx"], data.get("labels")
            )
            net.add_node(node, **attributes)

        # Add edges
        for edge in data["edge_index"]:
            source, target = edge
            net.add_edge(source, target, title=f"Edge {source} -> {target}")

        # Save if path provided
        if save_path:
            net.write_html(save_path)
            return save_path

        return net.html
