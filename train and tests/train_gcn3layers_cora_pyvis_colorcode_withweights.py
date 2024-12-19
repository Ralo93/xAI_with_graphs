import numpy as np
import torch
from torch_geometric.nn import GCNConv
from torch.nn import Linear, Dropout, ReLU
from torch_geometric.datasets import Planetoid
from torch_geometric.explain import Explainer, GNNExplainer
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network

class GCN(torch.nn.Module):
    def __init__(self, c_in, c_out, c_hidden):
        super().__init__()
        torch.manual_seed(12345)
        
        # Define the layers
        self.conv1 = GCNConv(c_in, c_hidden)  # Input to hidden layer
        self.conv2 = GCNConv(c_hidden, c_hidden)  # Hidden to hidden layer
        self.conv3 = GCNConv(c_hidden, c_out)  # Hidden to output layer
        
        self.dropout = Dropout(p=0.5, inplace=False)

    def forward(self, x, edge_index):
        # Pass through first layer
        x = self.conv1(x, edge_index)
        x = ReLU()(x)
        x = self.dropout(x)
        
        # Pass through second layer
        x = self.conv2(x, edge_index)
        x = ReLU()(x)
        x = self.dropout(x)
        
        # Pass through third layer (output layer)
        x = self.conv3(x, edge_index)
        
        return x

# Load dataset
dataset = Planetoid(root='data/Planetoid', name='Cora')
data = dataset[0]

# Train the GCN
c_in = dataset.num_features
c_out = dataset.num_classes
gcn_model = GCN(c_in=c_in, c_out=c_out, c_hidden=16)
loss_function = torch.nn.CrossEntropyLoss()  # Define loss function
optimizer = torch.optim.Adam(gcn_model.parameters(), lr=0.01, weight_decay=5e-4)  # Define optimizer

def train():
    gcn_model.train()
    optimizer.zero_grad()  # Clear gradients
    out = gcn_model(data.x, data.edge_index)  # Perform a single forward pass
    loss = loss_function(out[data.train_mask], data.y[data.train_mask])  # Compute the loss solely based on the training nodes
    loss.backward()  # Derive gradients
    optimizer.step()  # Update parameters based on gradients
    return loss

def test(mask):
    gcn_model.eval()  # Set model to evaluation mode
    out = gcn_model(data.x, data.edge_index)
    pred = out.argmax(dim=1)  # Use the class with the highest probability      
    correct = pred[mask] == data.y[mask]  # Check against ground-truth labels
    acc = int(correct.sum()) / int(mask.sum())  # Derive ratio of correct predictions
    return acc

# Training loop
for epoch in range(1, 201):
    train()
    train_acc = test(data.train_mask)
    val_acc = test(data.val_mask)
    test_acc = test(data.test_mask)
    print(
        f'Epoch: {epoch:03d}, Train acc: {train_acc:.4f}, '
        f'Val acc: {val_acc:.4f}, Test acc: {test_acc:.4f}'
    )

# prediction for target node
def predict_for_node(model, data, node_idx):
    """
    Predict the class label for a specific node.

    Args:
        model: The trained GCN model.
        data: The dataset (including graph data).
        node_idx: The index of the node to predict for.

    Returns:
        predicted_class: The predicted class label for the given node.
    """
    model.eval()  # Set model to evaluation mode
    out = model(data.x, data.edge_index)  # Get predictions for all nodes
    pred = out.argmax(dim=1)  # Predicted class for each node
    predicted_class = pred[node_idx].item()  # Get the predicted class for the specific node

    return predicted_class

# Now, run prediction for target node 
target_node_idx = int(input("Enter the node index you want to predict for: "))
predicted_label = predict_for_node(gcn_model, data, target_node_idx)

# Initialize the GNNExplainer and Explainer
explainer = Explainer(
    model=gcn_model, 
    algorithm=GNNExplainer(epochs=200), 
    explanation_type='model',
    node_mask_type='attributes',
    edge_mask_type='object',
    model_config=dict(
        mode='multiclass_classification', 
        task_level='node', 
        return_type='log_probs'
    )
)

# Generate the explanation for the node (node explanation)
explanation = explainer(data.x, data.edge_index, index=target_node_idx)

# Explanation will include the masked features and edge attributes
node_feat_mask, edge_mask = explanation.node_mask, explanation.edge_mask


def visualize_explanation_subgraph_with_pyvis(explanation, target_node_idx, data, threshold=0.5):
    """
    Visualize the subgraph explanation using Pyvis, including class labels, contributions for each node,
    and edge weights from the edge mask.
    
    Args:
        explanation: The explanation object with edge and node masks.
        target_node_idx: The index of the node being explained.
        data: The original graph data object.
        threshold: Importance score threshold for selecting edges (default is 0.5).
    """
    # Extract edge mask and edge index
    edge_mask = explanation.edge_mask.cpu().numpy()  # Edge importance scores
    edge_index = data.edge_index.cpu().numpy()

    # Filter edges with edge_mask values above the threshold
    important_edges = edge_index[:, edge_mask > threshold]
    important_edge_weights = edge_mask[edge_mask > threshold]

    # Convert edge weights to float64 to ensure compatibility with JSON serialization
    important_edge_weights = important_edge_weights.astype(float)

    # Create a Pyvis Network graph
    net = Network(height="800px", width="100%", bgcolor="#222222", font_color="white")
    net.set_options("""
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
    """)

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

    # Function to map node sizes and colors
    def get_node_size(node_idx):
        return 20 if node_idx == target_node_idx else 10
    
    def get_node_color(node_idx):
        # Target node is always black; others depend on their labels
        return "black" if node_idx == target_node_idx else label_colors.get(data.y[node_idx].item(), "white")

    # Add nodes with detailed labels
    node_set = set(important_edges.flatten())
    for node in node_set:
        # Get the label for the current node
        label = data.y[node].item()  # Assuming data.y contains node labels
        
        # Modify the target node label with additional information
        if node == target_node_idx:
            node_label = (
                f"Target Node (Predicted: {predicted_label}, Original: {label})"
            )
        else:
            node_label = f"Node {node}, Label: {label}"

        net.add_node(
            int(node), 
            label=node_label, 
            size=get_node_size(node),
            color=get_node_color(node),
            title=f"Node {int(node)}"
        )

    # Add edges with edge weights (from edge_mask)
    for idx, edge in enumerate(important_edges.T):
        src, dst = edge
        weight = important_edge_weights[idx]  # Weight for the edge
        net.add_edge(
            int(src), 
            int(dst), 
            value=weight,  # Sets the thickness of the edge
            title=f"Weight: {weight:.4f}"  # Hover text to display the weight
        )

    # Save and show the graph
    html_path = f"new_GCN_subgraph_node_{target_node_idx}.html"
    net.show(html_path, notebook=False)
    print(f"Subgraph visualization saved as {html_path}. Open it in your browser.")


# Call the function
visualize_explanation_subgraph_with_pyvis(explanation, target_node_idx, data, threshold=0.5)