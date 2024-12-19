import requests
from src.helpers.utils import *
from src.helpers.visualizing import *
from src.helpers.datasets import *

def test_predict_endpoint_re(target_node, num_hops, url, save):

    # Create a larger graph
    #node_features, edges = create_large_graph()

    re_dataset = Dataset('roman_empire')

    
    node_features= re_dataset.node_features
    edges = re_dataset.edges.t() # .t() because Cora...
    labels = re_dataset.labels

    target_label = labels[target_node]

    print(f"target label: {int(target_label)}")

    # Extract the subgraph for the target node
    input_data_dict = extract_subgraph(
        node_idx=target_node, num_hops=num_hops, node_features=node_features, edges=edges, labels=labels, take_all=False
    )

    #print("input_data_dict")
    #print(input_data_dict)

    #net = visualize_subgraph_pyvis(input_data_dict, save=True)

    # The subgraph extraction function now also has the target node we want to run prediction for
    target_node_idx = input_data_dict['target_node_idx']

    # get it out again because the request does not expect it
    del input_data_dict['target_node_idx']

    #try:
    print("Sending Subgraph to Prediction Endpoint...")
    response = requests.post(url, json=input_data_dict)

    # Check response status
    response.raise_for_status()

    # Parse the response
    result = response.json()
    
    print("Prediction Response Received.")
    print("what")
    print(result['edge_weights'])
    #print(result)

    #print(f"Result edge index: {result['edge_index']}")
    #print(f"Result edge weights for layers: {result['edge_weights']}")
    print(f"Class Probabilities for Target Node: {result['class_probabilities'][target_node_idx]}")

    probs_predicted = result['class_probabilities'][target_node_idx]

    net = visualize_subgraph_with_consistent_layout_re(result['edge_index'], result['edge_weights'], input_data_dict['labels'], target_node_idx, np.argmax(result['class_probabilities'][target_node_idx]), target_label, save=save)

    return result, target_node_idx

    #except requests.exceptions.RequestException as e:
    #    print(f"Error connecting to the endpoint: {e}")
    #except Exception as e:
    #    print(f"An error occurred: {e}")

def test_predict_endpoint_cora(target_node, num_hops, url, save):
    # Create a larger graph
    #node_features, edges = create_large_graph()

    cora_dataset = CoraDataset('data/cora.npz')

    node_features= cora_dataset.node_features
    edges = cora_dataset.edges.t() # .t() because Cora...
    labels = cora_dataset.labels

    target_label = labels[target_node]

    print(f"target label: {int(target_label)}")

    # Extract the subgraph for the target node
    input_data_dict = extract_subgraph(
        node_idx=target_node, num_hops=num_hops, node_features=node_features, edges=edges, labels=labels, take_all=False
    )

    #print("input_data_dict")
    #print(input_data_dict)

    #net = visualize_subgraph_pyvis(input_data_dict, save=True)

    # The subgraph extraction function now also has the target node we want to run prediction for
    target_node_idx = input_data_dict['target_node_idx']

    # get it out again because the request does not expect it
    del input_data_dict['target_node_idx']

    #try:
    print("Sending Subgraph to Prediction Endpoint...")
    response = requests.post(url, json=input_data_dict)

    # Check response status
    response.raise_for_status()

    # Parse the response
    result = response.json()
    
    print("Prediction Response Received.")


    #print(f"Result edge index: {result['edge_index']}")
    #print(f"Result edge weights for layers: {result['edge_weights']}")
    print(f"Class Probabilities for Target Node: {result['class_probabilities'][target_node_idx]}")

    probs_predicted = result['class_probabilities'][target_node_idx]

    net = visualize_subgraph_with_consistent_layout(result['edge_index'], result['edge_weights'], input_data_dict['labels'], target_node_idx, np.argmax(result['class_probabilities'][target_node_idx]), target_label, save=save)

    return result, target_node_idx

    #except requests.exceptions.RequestException as e:
    #    print(f"Error connecting to the endpoint: {e}")
    #except Exception as e:
    #    print(f"An error occurred: {e}")

# Run the test
if __name__ == "__main__":

    # URL of your local FastAPI endpoint
    URL_cora_3l = "http://localhost:8000/coGNN_predict_3layer_cora/"

    URL_cora_5l = "http://localhost:8000/coGNN_predict_5layer_cora/"

    URL_re_3l = "http://localhost:8000/coGNN_predict_3layer_re/"

    URL_re_5l = "http://localhost:8000/coGNN_predict_5layer_re/"

    URL_re_10l = "http://localhost:8000//coGNN_predict_10layer_re/"


    target_node = 210  # Take 1000 as a good example NODE 5, 10 almost complete isolation but not working well. Node 21 for information flow explanation
    result, target_node_idx = test_predict_endpoint_re(target_node, num_hops=10, url=URL_re_10l, save=True)
    target_class_probabilities = result['class_probabilities'][target_node_idx]
    predicted_class = target_class_probabilities.index(max(target_class_probabilities))
    print(f"Predicted class for target node: {predicted_class}")

