import requests
from src.helpers.utils import *
from src.helpers.visualizing import *
from src.helpers.datasets import *

# URL of your local FastAPI endpoint
URL = "http://localhost:8000/predict/"

#CLOUD_URL = "https://cora-gat-image-196616273613.europe-west10.run.app/predict/"


def test_predict_endpoint(target_node, num_hops=3):
    # Create a larger graph
    #node_features, edges = create_large_graph()

    cora_dataset = CoraDataset('data/cora.npz')

    node_features= cora_dataset.node_features
    edges = cora_dataset.edges.t() # .t() because Cora...
    labels = cora_dataset.labels

    target_label = labels[target_node]

    print(int(target_label))

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

    visualize_analysis_with_layers(input_data_dict, normalized_analysis, target_node_idx)

    return result, target_node_idx

    #except requests.exceptions.RequestException as e:
    #    print(f"Error connecting to the endpoint: {e}")
    #except Exception as e:
    #    print(f"An error occurred: {e}")



# Run the test
if __name__ == "__main__":

    target_node = 5  # Specify the node to predict
    result, target_node_idx = test_predict_endpoint(target_node)
    target_class_probabilities = result['class_probabilities'][target_node_idx]
    predicted_class = target_class_probabilities.index(max(target_class_probabilities))
    print(f"Predicted class for target node: {predicted_class}")

