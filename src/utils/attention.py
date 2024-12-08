from typing import Dict, List, Any
import numpy as np


class AttentionAnalyzer:
    """Analyzer for attention weights in graph neural networks"""

    @staticmethod
    def softmax_normalize(values: np.ndarray) -> np.ndarray:
        """Normalize values using softmax"""
        exp_values = np.exp(values)
        return exp_values / np.sum(exp_values)

    @staticmethod
    def analyze_attention_weights(result: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze multi-layer attention weights"""
        attention_weights = result.get("attention_weights", [])
        if not attention_weights:
            raise ValueError("No attention weights found in results")

        edge_index = np.array(result["edge_index"])
        num_nodes = np.max(edge_index) + 1

        analysis = AttentionAnalyzer._initialize_analysis()
        total_sum = np.zeros(len(edge_index[0]))

        # Analyze each layer
        for layer_num, layer_weights in enumerate(attention_weights, 1):
            layer_weights = np.array(layer_weights)
            layer_sum = layer_weights.sum(axis=1)
            total_sum += layer_sum

            layer_results = AttentionAnalyzer._analyze_layer(
                layer_num, layer_sum, edge_index, num_nodes
            )
            analysis["layers"].append(layer_results)

        # Analyze overall results
        analysis["overall"] = AttentionAnalyzer._analyze_overall(
            total_sum, edge_index, num_nodes
        )

        return analysis

    @staticmethod
    def _initialize_analysis() -> Dict[str, Any]:
        """Initialize analysis results structure"""
        return {"layers": [], "overall": {"top_edges": [], "top_nodes": []}}

    @staticmethod
    def _analyze_layer(
        layer_num: int, layer_sum: np.ndarray, edge_index: np.ndarray, num_nodes: int
    ) -> Dict[str, Any]:
        """Analyze attention weights for a single layer"""
        node_contributions = AttentionAnalyzer._calculate_node_contributions(
            layer_sum, edge_index, num_nodes
        )

        normalized_edge_weights = AttentionAnalyzer.softmax_normalize(layer_sum)
        normalized_node_contribs = AttentionAnalyzer.softmax_normalize(
            node_contributions
        )

        return {
            "layer_number": layer_num,
            "top_edges": AttentionAnalyzer._get_top_edges(
                edge_index, layer_sum, normalized_edge_weights
            ),
            "top_nodes": AttentionAnalyzer._get_top_nodes(
                node_contributions, normalized_node_contribs
            ),
        }

    @staticmethod
    def _analyze_overall(
        total_sum: np.ndarray, edge_index: np.ndarray, num_nodes: int
    ) -> Dict[str, List]:
        """Analyze overall attention patterns"""
        node_contributions = AttentionAnalyzer._calculate_node_contributions(
            total_sum, edge_index, num_nodes
        )

        normalized_total = AttentionAnalyzer.softmax_normalize(total_sum)
        normalized_node_contribs = AttentionAnalyzer.softmax_normalize(
            node_contributions
        )

        return {
            "top_edges": AttentionAnalyzer._get_top_edges(
                edge_index, total_sum, normalized_total
            ),
            "top_nodes": AttentionAnalyzer._get_top_nodes(
                node_contributions, normalized_node_contribs
            ),
        }

    @staticmethod
    def _calculate_node_contributions(
        weights: np.ndarray, edge_index: np.ndarray, num_nodes: int
    ) -> np.ndarray:
        """Calculate contribution of each node based on edge weights"""
        contributions = np.zeros(num_nodes)
        for (src, dest), weight in zip(edge_index.T, weights):
            contributions[src] += weight
            contributions[dest] += weight
        return contributions

    @staticmethod
    def _get_top_edges(
        edge_index: np.ndarray,
        weights: np.ndarray,
        normalized_weights: np.ndarray,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Get top-k edges by weight"""
        top_indices = weights.argsort()[::-1][:top_k]
        return [
            {
                "index": idx,
                "source": edge_index[0][idx],
                "target": edge_index[1][idx],
                "normalized_weight": normalized_weights[idx],
            }
            for idx in top_indices
        ]

    @staticmethod
    def _get_top_nodes(
        contributions: np.ndarray, normalized_contributions: np.ndarray, top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Get top-k nodes by contribution"""
        top_indices = contributions.argsort()[::-1][:top_k]
        return [
            {"index": idx, "normalized_contribution": normalized_contributions[idx]}
            for idx in top_indices
        ]
