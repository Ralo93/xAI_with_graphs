import pytest
import requests
from src.utils.graph import GraphUtils
from src.visualization.layer_vis import LayerVisualizer


class TestPredictionEndpoint:
    URL = "http://localhost:8000/predict/"

    def test_predict_endpoint(self, cora_dataset):
        """Test the prediction endpoint with Cora dataset"""
        # Prepare test data
        target_node = 1000  # Example node
        num_hops = 3

        # Extract subgraph
        graph_utils = GraphUtils()
        input_data = graph_utils.extract_subgraph(
            node_idx=target_node,
            num_hops=num_hops,
            node_features=cora_dataset.node_features,
            edges=cora_dataset.edges.t(),
            labels=cora_dataset.labels,
        )

        # Create visualization
        visualizer = LayerVisualizer()
        visualizer.visualize(input_data, save_path="test_subgraph.html")

        # Store target info
        target_node_idx = input_data.pop("target_node_idx")
        target_label = cora_dataset.labels[target_node]

        # Make prediction request
        response = requests.post(self.URL, json=input_data)
        assert response.status_code == 200

        result = response.json()

        # Verify response structure
        assert "class_probabilities" in result
        assert "attention_weights" in result

        # Get predicted class
        probs = result["class_probabilities"][target_node_idx]
        predicted_class = max(range(len(probs)), key=probs.__getitem__)

        # Visualize results
        from src.utils.attention import AttentionAnalyzer

        analyzer = AttentionAnalyzer()
        analysis = analyzer.analyze_attention_weights(result)

        visualizer = LayerVisualizer()
        visualizer.visualize(
            {
                "subgraph_data": input_data,
                "analysis_results": analysis,
                "target": target_node_idx,
                "predicted_class": predicted_class,
                "target_label": target_label,
            },
            save_dir="test_visualizations",
        )
