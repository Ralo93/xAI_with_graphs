from abc import ABC, abstractmethod
import torch
from typing import Tuple, Optional
import numpy as np


class BaseDataset(ABC):
    """Abstract base class for all graph datasets"""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.node_features = None
        self.labels = None
        self.edges = None

    @abstractmethod
    def load_data(self) -> None:
        """Load dataset from source"""
        pass

    def to_device(self) -> None:
        """Move dataset to specified device"""
        if self.node_features is not None:
            self.node_features = self.node_features.to(self.device)
        if self.labels is not None:
            self.labels = self.labels.to(self.device)
        if self.edges is not None:
            self.edges = self.edges.to(self.device)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__} with "
            f"{self.node_features.size(0) if self.node_features is not None else 0} nodes, "
            f"{self.edges.size(0) if self.edges is not None else 0} edges, and "
            f"{len(torch.unique(self.labels)) if self.labels is not None else 0} classes."
        )
