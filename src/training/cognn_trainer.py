from pathlib import Path
from models.model_registry import ModelRegistry
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from dataclasses import dataclass
from typing import Optional
import logging
import mlflow
from ..models.cognn import CoGNN, CoGNNConfig


@dataclass
class CoGNNTrainingConfig:
    """Training configuration for CoGNN"""

    LEARNING_RATE: float = 0.001
    WEIGHT_DECAY: float = 0
    EPOCHS: int = 200
    BATCH_SIZE: int = 32
    DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CoGNNTrainer:
    """Trainer for CoGNN model"""

    def __init__(
        self,
        config: CoGNNTrainingConfig = CoGNNTrainingConfig(),
        logger: Optional[logging.Logger] = None,
    ):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.device = config.DEVICE

    def train(self, model: CoGNN, data):
        """
        Train the CoGNN model

        Args:
            model: CoGNN model instance
            data: PyTorch Geometric data object with train/val/test masks

        Returns:
            float: Best validation accuracy
        """
        data = data.to(self.device)
        model = model.to(self.device)

        optimizer = Adam(
            model.parameters(),
            lr=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY,
        )
        criterion = CrossEntropyLoss()

        best_val_accuracy = 0.0

        for epoch in range(self.config.EPOCHS):
            # Training
            model.train()
            optimizer.zero_grad()

            out, _ = model(data.x, data.edge_index)
            train_loss = criterion(out[data.train_mask], data.y[data.train_mask])

            train_loss.backward()
            optimizer.step()

            # Validation
            model.eval()
            with torch.no_grad():
                out, _ = model(data.x, data.edge_index)
                val_loss = criterion(out[data.val_mask], data.y[data.val_mask])

                # Calculate accuracies
                train_acc = self._calculate_accuracy(
                    out[data.train_mask], data.y[data.train_mask]
                )
                val_acc = self._calculate_accuracy(
                    out[data.val_mask], data.y[data.val_mask]
                )

                # Log metrics
                metrics = {
                    "train_loss": train_loss.item(),
                    "val_loss": val_loss.item(),
                    "train_accuracy": train_acc,
                    "val_accuracy": val_acc,
                }

                mlflow.log_metrics(metrics, step=epoch)

                # Save best model
                if val_acc > best_val_accuracy:
                    best_val_accuracy = val_acc
                    mlflow.pytorch.log_model(model, "model")
                    # Save the model locally
                    ModelRegistry.save_model(model, Path("models/artifacts/model.pth"))

            if epoch % 10 == 0:
                self.logger.info(
                    f"Epoch {epoch}/{self.config.EPOCHS}: "
                    f"Train Loss: {train_loss:.4f}, "
                    f"Val Loss: {val_loss:.4f}, "
                    f"Train Acc: {train_acc:.4f}, "
                    f"Val Acc: {val_acc:.4f}"
                )

        return best_val_accuracy

    def _calculate_accuracy(self, output: torch.Tensor, target: torch.Tensor) -> float:
        """Calculate classification accuracy"""
        pred = output.argmax(dim=1)
        correct = (pred == target).sum().item()
        total = target.size(0)
        return correct / total
