import requests
from src.helpers.utils import *
from src.helpers.visualizing import *
from src.helpers.datasets import *

# URL of your local FastAPI endpoint
URL = "http://localhost:8000/gat_predict_cora/"


#CLOUD_URL = "https://cora-gat-image-196616273613.europe-west10.run.app/predict/"


def test_predict_endpoint(target_node, num_hops=3):
    # Create a larger graph
    #node_features, edges = create_large_graph()

    cora_dataset = CoraDataset('data/cora.npz')

    node_features= cora_dataset.node_features
    edges = cora_dataset.edges.t() # .t() because Cora...
    labels = cora_dataset.labels

    target_label = labels[target_node]

    print(f"target label: {int(target_label)}")

    # Extract the subgraph for the target node
    input_data_dict_subgraph = extract_subgraph(
        node_idx=target_node, num_hops=num_hops, node_features=node_features, edges=edges, labels=labels, take_all=False
    )
    print("shapes")
    print(type(input_data_dict_subgraph))  # To confirm it's a dictionary
    print(input_data_dict_subgraph.keys())  # To see what keys the dictionary contains

        # Extract the subgraph for the target node
    #input_data_dict_all = extract_subgraph(
    #    node_idx=target_node, num_hops=num_hops, node_features=node_features, edges=edges, labels=labels, take_all=True
    #)
    #print(type(input_data_dict_all))  # To confirm it's a dictionary
    #print(input_data_dict_all.keys())  # To see what keys the dictionary contains

#    net = visualize_subgraph_pyvis(input_data_dict, save=True)

    # The subgraph extraction function now also has the target node we want to run prediction for
    target_node_idx_subgraph = input_data_dict_subgraph['target_node_idx']
    #target_node_idx_all = input_data_dict_all['target_node_idx']

    #assert target_node_idx == target_node_idx_all

    # get it out again because the request does not expect it
    del input_data_dict_subgraph['target_node_idx']
    #del input_data_dict_all['target_node_idx']

    #try:
    print("Sending Subgraph to Prediction Endpoint...")
    response = requests.post(URL, json=input_data_dict_subgraph)

    # Check response status
    response.raise_for_status()
        # Check response status
    #response_all.raise_for_status()

    # Parse the response
    result_subgraph = response.json()
        # Parse the response

    #result_all = response_all.json()
    
    print("Prediction Response Received.")

    # Extract and process attention weights
    aw_subgraph = result_subgraph.get('attention_weights', [])
    #print(f"Result subgraph: {aw_subgraph}")

    # Convert to NumPy array for shape inspection
    array_data_sub = np.array(aw_subgraph)

    # Get the shape
    print("Shape of the data for SUB:", array_data_sub.shape)

    # Extract and process attention weights
    #aw_all = result_all.get('attention_weights', [])
   # #print(f"Result all: {aw_all}")
        # Convert to NumPy array for shape inspection
    #array_data_all = np.array(aw_all)

    # Get the shape
    #print("Shape of the data for ALL:", array_data_all.shape)

    # Flatten both arrays to compare element-wise
    array_data_flat = array_data_sub.flatten()
    #another_array_flat = array_data_all.flatten()

    # Check if all elements in `array_data` are in `another_array`
    #is_subset = np.all(np.isin(array_data_flat, another_array_flat))

    #print("Is array_data a subset of another_array?", is_subset)

    print(f"Class Probabilities for Target Node SUB: {result_subgraph['class_probabilities'][target_node_idx_subgraph]}")
    #print(f"Class Probabilities for Target Node ALL: {result_all['class_probabilities'][target_node_idx_all]}")

    probs_predicted_sub = result_subgraph['class_probabilities'][target_node_idx_subgraph]
    #probs_predicted_all = result_all['class_probabilities'][target_node_idx_all]
    
    #normalized_analysis_subgraph = analyze_attention_weights(result_subgraph)
    normalized_analysis_subgraph = analyze_attention_weights(result_subgraph)

    #print(normalized_analysis_subgraph)
    #normalized_analysis_all = analyze_attention_weights(result_all)
    #print(normalized_analysis_all)

    visualize_analysis_with_layers(input_data_dict_subgraph, normalized_analysis_subgraph, target_node_idx_subgraph, np.argmax(probs_predicted_sub), target_label)

    return result_subgraph, target_node_idx_subgraph

    #except requests.exceptions.RequestException as e:
    #    print(f"Error connecting to the endpoint: {e}")
    #except Exception as e:
    #    print(f"An error occurred: {e}")



# Run the test
if __name__ == "__main__":

    target_node = 1000  # Take 1000 as a good example
    result, target_node_idx = test_predict_endpoint(target_node)
    target_class_probabilities = result['class_probabilities'][target_node_idx]
    predicted_class = target_class_probabilities.index(max(target_class_probabilities))
    print(f"Predicted class for target node: {predicted_class}")



    