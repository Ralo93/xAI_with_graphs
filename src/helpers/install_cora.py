from torch_geometric.datasets import Planetoid
import os
import numpy as np
import scipy.sparse as sp

try:
    # Load the Cora dataset
    dataset = Planetoid(root='~/somewhere/Cora', name='Cora')
    data = dataset[0]

    # Prepare the directory to save the dataset
    save_dir = "data"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "cora.npz")

    # Convert the data to a format compatible with .npz
    # Adjacency matrix
    row, col = data.edge_index.numpy()
    adjacency_matrix = sp.coo_matrix((np.ones(len(row)), (row, col)), shape=(data.num_nodes, data.num_nodes))

    # Feature matrix
    feature_matrix = data.x.numpy()

    # Labels
    labels = data.y.numpy()

    # Save the matrices into .npz
    np.savez(save_path, 
            adjacency=sp.csr_matrix(adjacency_matrix), 
            features=feature_matrix, 
            labels=labels)

    print(f"Successfully downloaded cora at {save_path}")

except:
    print("Problem in downloading and saving cora!")
