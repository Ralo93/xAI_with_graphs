from torch_geometric.datasets import Planetoid
import os
import numpy as np
import scipy.sparse as sp
from pathlib import Path


def download_cora():
    try:
        # Get the project root directory (2 levels up from this file)
        root_dir = Path(__file__).parent.parent.parent

        # Define data directories
        raw_dir = root_dir / "data" / "raw"

        # Create directories if they don't exist
        raw_dir.mkdir(parents=True, exist_ok=True)

        # Define save path
        save_path = raw_dir / "cora.npz"

        # Load the Cora dataset
        dataset = Planetoid(root="~/somewhere/Cora", name="Cora")
        data = dataset[0]

        # Convert the data to a format compatible with .npz
        # Adjacency matrix
        row, col = data.edge_index.numpy()
        adjacency_matrix = sp.coo_matrix(
            (np.ones(len(row)), (row, col)), shape=(data.num_nodes, data.num_nodes)
        )

        # Feature matrix
        feature_matrix = data.x.numpy()

        # Labels
        labels = data.y.numpy()

        # Save the matrices into .npz
        np.savez(
            save_path,
            adjacency=sp.csr_matrix(adjacency_matrix),
            features=feature_matrix,
            labels=labels,
        )

        print(f"Successfully downloaded Cora dataset at {save_path}")

    except Exception as e:
        print(f"Problem in downloading and saving Cora: {e}")
        raise


if __name__ == "__main__":
    download_cora()
