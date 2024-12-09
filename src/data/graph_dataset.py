from .base import BaseDataset
import os
import torch
import numpy as np
from typing import Optional


class GraphDataset(BaseDataset):
    """Generic graph dataset loader"""

    def __init__(
        self,
        name: str,
        data_dir: str = "data",
        add_self_loops: bool = False,
        device: str = "cpu",
        use_sgc_features: bool = False,
        use_identity_features: bool = False,
        use_adjacency_features: bool = False,
        do_not_use_original_features: bool = False,
    ):
        super().__init__(device)
        self.name = name
        self.data_dir = data_dir
        self.add_self_loops = add_self_loops
        self.feature_flags = {
            "sgc": use_sgc_features,
            "identity": use_identity_features,
            "adjacency": use_adjacency_features,
            "original": not do_not_use_original_features,
        }
        self._validate_features()
        self.load_data()

    def _validate_features(self) -> None:
        """Validate feature configuration"""
        if not self.feature_flags["original"] and not any(
            [
                self.feature_flags["sgc"],
                self.feature_flags["identity"],
                self.feature_flags["adjacency"],
            ]
        ):
            raise ValueError(
                "If original node features are not used, at least one of "
                "[use_sgc_features, use_identity_features, use_adjacency_features] "
                "should be True."
            )

    def load_data(self) -> None:
        """Load dataset from NPZ file"""
        file_path = os.path.join(self.data_dir, f'{self.name.replace("-", "_")}.npz')
        data = np.load(file_path)

        self.node_features = torch.tensor(data["node_features"], dtype=torch.float32)
        self.labels = torch.tensor(data["node_labels"], dtype=torch.long)
        self.edges = torch.tensor(data["edges"], dtype=torch.long)

        self.to_device()
