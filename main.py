from src.helpers.datasets import *
from tensorboard.plugins.projector import ProjectorConfig, visualize_embeddings
import tensorflow as tf
import numpy as np
import torch
import os


def main():
    # Load the Cora dataset
    cora_dataset = CoraDataset('data/cora.npz')
    print(cora_dataset)

    # Extract data
    nodes = cora_dataset.node_features.numpy()  # Convert PyTorch tensor to NumPy
    labels = cora_dataset.labels.numpy()        # Convert PyTorch tensor to NumPy

    
    edges = cora_dataset.edges

    # Ensure the logs directory exists
    os.makedirs("logs", exist_ok=True)

    # Save embeddings to TensorFlow checkpoint
    embedding_var = tf.Variable(nodes, name='my_embeddings')
    checkpoint = tf.train.Checkpoint(embedding=embedding_var)
    checkpoint.save("logs/embedding.ckpt")

    # Get unique labels and their corresponding names
    unique_labels = np.unique(labels)
    label_names = cora_dataset.label_names if hasattr(cora_dataset, 'label_names') else [f"Class {label}" for label in unique_labels]
    
    # Create a label mapping dictionary
    label_map = {label: name for label, name in zip(unique_labels, label_names)}

    # Save metadata to a .tsv file with more detailed information
    with open("logs/metadata.tsv", "w") as f:
        # Write header
        f.write("Index\tLabel\tLabel Name\n")
        
        # Write data rows
        for idx, label in enumerate(labels):
            label_name = label_map.get(label, f"Unknown Class {label}")
            f.write(f"{idx}\t{label}\t{label_name}\n")

    # Create a log directory for TensorBoard
    log_dir = "logs"

    # Initialize the projector config
    config = ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = embedding_var.name  # TensorFlow variable name
    embedding.metadata_path = "metadata.tsv"   # Metadata file

    # Save the projector config file
    visualize_embeddings(log_dir, config)

    print("TensorBoard visualization files have been generated in the 'logs' directory.")


# The standard Python entry point
if __name__ == "__main__":
    main()