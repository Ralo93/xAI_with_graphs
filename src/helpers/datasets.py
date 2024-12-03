import os
import numpy as np
import torch
import scipy.sparse as sp


# use this for roman-empire, questions or any other dataset but not cora
class Dataset:
    def __init__(self, name, add_self_loops=False, device='cpu', use_sgc_features=False, use_identity_features=False,
                 use_adjacency_features=False, do_not_use_original_features=False):

        if do_not_use_original_features and not any([use_sgc_features, use_identity_features, use_adjacency_features]):
            raise ValueError('If original node features are not used, at least one of the arguments '
                             'use_sgc_features, use_identity_features, use_adjacency_features should be used.')

        print('Preparing data...')
        self.data = np.load(os.path.join('data', f'{name.replace("-", "_")}.npz'))
        self.node_features = torch.tensor(self.data['node_features'])
        self.labels = torch.tensor(self.data['node_labels'])
        self.edges = torch.tensor(self.data['edges'])
        
        return
    

class CoraDataset:
    def __init__(self, npz_file, add_self_loops=False, device='cpu'):
        """
        Initialize the Cora dataset from a .npz file.
        
        Args:
            npz_file (str): Path to the .npz file
            add_self_loops (bool): Whether to add self-loops to the adjacency matrix
            device (str): Device to load tensors on
        """
        # Load the .npz file
        data = np.load(npz_file, allow_pickle=True)
        
        # Print available keys for debugging
        print("Available keys in .npz file:", list(data.keys()))
        
        # Load node features
        node_features = data['features']
        
        # Load labels
        labels = data['labels']
        
        # Handle adjacency matrix
        adjacency = data['adjacency']

        print("Adjacency type:", type(adjacency))
        print("Adjacency dtype:", adjacency.dtype)
        if hasattr(adjacency, 'shape'):
            print("Adjacency shape:", adjacency.shape)
        
        # Special handling for scipy sparse matrix
        if isinstance(adjacency, np.ndarray) and adjacency.dtype == object:
            # This suggests the adjacency matrix is a scipy sparse matrix stored as an object
            adjacency = adjacency.item()
        
        # Convert adjacency to edge index
        if sp.issparse(adjacency):
            adjacency = adjacency.tocoo()
            row, col = adjacency.row, adjacency.col
        elif isinstance(adjacency, np.ndarray):
            if adjacency.ndim == 2:
                # 2D adjacency matrix
                row, col = np.nonzero(adjacency)
            elif adjacency.ndim == 1:
                # Edge list format
                row, col = adjacency[0], adjacency[1]
            else:
                raise ValueError(f"Unexpected adjacency matrix format. Dimensions: {adjacency.ndim}")
        else:
            raise ValueError(f"Unsupported adjacency matrix type: {type(adjacency)}")
        
        # Add self-loops if requested
        if add_self_loops:
            num_nodes = node_features.shape[0]
            self_loop_row = np.arange(num_nodes)
            self_loop_col = np.arange(num_nodes)
            row = np.concatenate([row, self_loop_row])
            col = np.concatenate([col, self_loop_col])
        
        # Convert to PyTorch tensors
        self.node_features = torch.tensor(node_features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.edges = torch.tensor([row, col], dtype=torch.long).t() #cora needs transposition, because data is in a slightly different representation than e.g. roman-empire
        
        # Move to specified device
        self.node_features = self.node_features.to(device)
        self.labels = self.labels.to(device)
        self.edges = self.edges.to(device)
    
    def __repr__(self):
        """
        String representation of the dataset
        """
        return (f"CoraDataset with {self.node_features.size(0)} nodes, "
                f"{self.edges.size(0)} edges, and {len(torch.unique(self.labels))} classes.")
       