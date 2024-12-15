import numpy as np
import torch
from torch_geometric.nn import GCNConv
from torch.nn import Linear, Dropout, ReLU
from torch_geometric.datasets import Planetoid
from torch_geometric.explain import Explainer, GNNExplainer
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

# GAT model
class GAT(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 num_heads: int = 4, dropout: float = 0.3, edge_dim: int = None):
        super().__init__()

        self.dropout = Dropout(dropout)

        # Calculate output dimensions
        self.gat1_out_channels = hidden_channels * num_heads
        self.gat2_out_channels = hidden_channels * num_heads

        # Replace BatchNorm with LayerNorm
        self.norm1 = nn.LayerNorm(self.gat1_out_channels)
        self.norm2 = nn.LayerNorm(self.gat2_out_channels)

        # Input projection for skip connection
        self.proj_skip = nn.Linear(in_channels, self.gat2_out_channels)

        self.gat1 = GATConv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            heads=num_heads,
            concat=True,
            dropout=dropout,
            edge_dim=edge_dim,
            add_self_loops=False  ## The underlaying model might have self loops!!
        )

        self.gat2 = GATConv(
            in_channels=hidden_channels * num_heads,
            out_channels=hidden_channels,
            heads=num_heads,
            concat=True,
            dropout=dropout,
            add_self_loops=False
        )

        self.gat3 = GATConv(
            in_channels=hidden_channels * num_heads,
            out_channels=out_channels,# // num_heads,
            heads=num_heads,
            concat=False,
            dropout=dropout,
            add_self_loops=False
        )

    def forward(self, x, edge_index, return_attention=False):
        """Debugging: return logits and attention weights."""
        # Save input for skip connection
        x_skip = x

        # GAT layer 1
        x, alpha1 = self.gat1(x, edge_index, return_attention_weights=True)
        x = F.elu(x)
        x = self.norm1(x)
        x = self.dropout(x)

        # GAT layer 2
        x_skip = self.proj_skip(x_skip)  # Align dimensions for skip connection
        x = x + x_skip  # Add skip connection
        x, alpha2 = self.gat2(x, edge_index, return_attention_weights=True)
        x = F.elu(x)
        x = self.norm2(x)
        x = self.dropout(x)

        # GAT layer 3
        x, alpha3 = self.gat3(x, edge_index, return_attention_weights=True)

        if return_attention:
            return x, [alpha1, alpha2, alpha3]
        return x
    


# Load dataset
dataset = Planetoid(root='data/Planetoid', name='Cora')
data = dataset[0]

# Train the GAT

# Define model, loss function, and optimizer
c_in = dataset.num_features
c_out = dataset.num_classes
gat_model = GAT(in_channels=c_in, hidden_channels=16, out_channels=c_out)
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(gat_model.parameters(), lr=0.01, weight_decay=5e-4)

def train():
    gat_model.train()
    optimizer.zero_grad()
    out = gat_model(data.x, data.edge_index)  # Include attention outputs
    loss = loss_function(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def test(mask):
    gat_model.eval()
    with torch.no_grad():  # Disable gradient computation
        out = gat_model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        correct = (pred[mask] == data.y[mask]).sum().item()
        total = mask.sum().item()
        return correct / total

for epoch in range(1, 10):
    train_loss = train()
    train_acc = test(data.train_mask)
    val_acc = test(data.val_mask)
    test_acc = test(data.test_mask)
    print(
        f"Epoch: {epoch:03d}, Loss: {train_loss:.4f}, "
        f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}"
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
predicted_label = predict_for_node(gat_model, data, target_node_idx)

# Initialize the GNNExplainer and Explainer
explainer = Explainer(
    model=gat_model, 
    algorithm=GNNExplainer(epochs=10), 
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
        return "red" if node_idx == target_node_idx else label_colors.get(data.y[node_idx].item(), "white")

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
    html_path = f"new_GAT_subgraph_node_{target_node_idx}.html"
    net.show(html_path, notebook=False)
    print(f"Subgraph visualization saved as {html_path}. Open it in your browser.")


# Call the function
visualize_explanation_subgraph_with_pyvis(explanation, target_node_idx, data, threshold=0.2)