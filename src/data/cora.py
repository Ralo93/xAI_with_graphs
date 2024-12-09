from .base import BaseDataset
import torch
import numpy as np
import scipy.sparse as sp
from typing import Optional, Tuple


class CoraDataset(BaseDataset):
    """Cora citation network dataset loader"""

    def __init__(
        self, npz_file: str, add_self_loops: bool = False, device: str = "cpu"
    ):
        super().__init__(device)
        self.npz_file = npz_file
        self.add_self_loops = add_self_loops
        self.load_data()

    def _process_adjacency(self, adjacency) -> Tuple[np.ndarray, np.ndarray]:
        """Process adjacency matrix into edge indices"""
        if isinstance(adjacency, np.ndarray) and adjacency.dtype == object:
            adjacency = adjacency.item()

        if sp.issparse(adjacency):
            adjacency = adjacency.tocoo()
            row, col = adjacency.row, adjacency.col
        elif isinstance(adjacency, np.ndarray):
            if adjacency.ndim == 2:
                row, col = np.nonzero(adjacency)
            elif adjacency.ndim == 1:
                row, col = adjacency[0], adjacency[1]
            else:
                raise ValueError(
                    f"Unexpected adjacency matrix format. Dimensions: {adjacency.ndim}"
                )
        else:
            raise ValueError(f"Unsupported adjacency matrix type: {type(adjacency)}")

        return row, col

    def _add_self_loops(
        self, row: np.ndarray, col: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Add self-loops to the edge indices"""
        if self.add_self_loops:
            num_nodes = self.node_features.shape[0]
            self_loop_row = np.arange(num_nodes)
            self_loop_col = np.arange(num_nodes)
            row = np.concatenate([row, self_loop_row])
            col = np.concatenate([col, self_loop_col])
        return row, col

    def load_data(self) -> None:
        """Load Cora dataset from NPZ file"""
        data = np.load(self.npz_file, allow_pickle=True)

        # Load features and labels
        self.x = torch.tensor(data["features"], dtype=torch.float32)
        self.y = torch.tensor(data["labels"], dtype=torch.long)

        # Process adjacency matrix
        row, col = self._process_adjacency(data["adjacency"])
        row, col = self._add_self_loops(row, col)

        # Create edge index tensor
        self.edge_index = torch.tensor([row, col], dtype=torch.long)

        # Create data splits
        self._create_masks()

        # Move to device
        self.to_device()

    def _create_masks(self, train_ratio=0.8, val_ratio=0.1):
        """Create train/val/test masks"""
        num_nodes = self.x.size(0)
        indices = torch.randperm(num_nodes)

        train_size = int(train_ratio * num_nodes)
        val_size = int(val_ratio * num_nodes)

        self.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        self.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        self.test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        self.train_mask[indices[:train_size]] = True
        self.val_mask[indices[train_size : train_size + val_size]] = True
        self.test_mask[indices[train_size + val_size :]] = True

    def to(self, device):
        """Move dataset to specified device"""
        self.edge_index = self.edge_index.to(device)
        self.x = self.x.to(device)
        self.y = self.y.to(device)
        if hasattr(self, "train_mask"):
            self.train_mask = self.train_mask.to(device)
            self.val_mask = self.val_mask.to(device)
            self.test_mask = self.test_mask.to(device)
        return self

    def to_device(self) -> None:
        """Move dataset to specified device"""
        if self.x is not None:
            self.x = self.x.to(self.device)
            self.node_features = self.x
        if self.y is not None:
            self.y = self.y.to(self.device)
            self.labels = self.y
        if self.edge_index is not None:
            self.edge_index = self.edge_index.to(self.device)
            self.edges = self.edge_index.t()

    def __repr__(self) -> str:
        return (
            f"CoraDataset with {self.x.size(0)} nodes, "
            f"{self.edge_index.size(1)} edges, and "
            f"{len(torch.unique(self.y))} classes."
        )
