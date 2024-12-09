from src.data.cora import CoraDataset
from src.models.gat import GAT, GATConfig
from src.models.model_registry import ModelRegistry
from src.visualization.base import BaseVisualizer
import numpy as np
import os
from pathlib import Path
from src.config.settings import settings


def main():
    # Load the Cora dataset
    cora_dataset = CoraDataset(
        "data/raw/cora.npz"
    )  # Updated path to match new structure

    # Extract data
    nodes_features = (
        cora_dataset.node_features.numpy()
    )  # Convert PyTorch tensor to NumPy
    labels = cora_dataset.labels.numpy()  # Convert PyTorch tensor to NumPy
    edges = cora_dataset.edges

    # Load the trained model using ModelRegistry
    config = GATConfig(
        in_channels=1433,  # Cora features dimension
        hidden_channels=5,  # Hidden layer size
        out_channels=7,  # Number of classes
    )
    model = ModelRegistry.load_model(config, model_path=settings.MODEL_PATH)

    # Do 1 forward pass with the model
    outputs = model(cora_dataset.node_features, edges)

    # Create logs directory in the project structure
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Get unique labels and their corresponding names
    unique_labels = np.unique(labels)
    label_names = (
        cora_dataset.label_names
        if hasattr(cora_dataset, "label_names")
        else [f"Class {label}" for label in unique_labels]
    )

    # Create a label mapping dictionary
    label_map = {label: name for label, name in zip(unique_labels, label_names)}

    # Save metadata to a .tsv file with more detailed information
    metadata_path = log_dir / "metadata.tsv"
    with open(metadata_path, "w") as f:
        # Write header
        f.write("Index\tLabel\tLabel Name\n")

        # Write data rows
        for idx, label in enumerate(labels):
            label_name = label_map.get(label, f"Unknown Class {label}")
            f.write(f"{idx}\t{label}\t{label_name}\n")

    print("Metadata files have been generated in the 'logs' directory.")


# The standard Python entry point
if __name__ == "__main__":
    main()
