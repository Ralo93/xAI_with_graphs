import torch
import mlflow
import logging
from typing import Optional
from ..models.gat import GAT
from ..utils.metrics import MetricsCalculator
from ..utils.graph import GraphUtils
from ..config.settings import Settings


class GATTrainer:
    """Trainer class for Graph Attention Networks"""

    def __init__(self, config: Settings, logger: Optional[logging.Logger] = None):
        self.config = config
        self.device = config.device
        self.logger = logger or logging.getLogger(__name__)
        self.metrics = MetricsCalculator()
        self.graph_utils = GraphUtils()

    def prepare_data(self, data):
        """Prepare data for training"""
        # Add self-loops and make bidirectional
        edge_index = self.graph_utils.make_bidirectional(data.edge_index)

        # Create train/val/test masks
        num_nodes = data.x.size(0)
        shuffled_indices = torch.randperm(num_nodes)

        num_train = int(self.config.TRAIN_SPLIT * num_nodes)
        num_val = int(self.config.VAL_SPLIT * num_nodes)

        masks = {
            "train": torch.zeros(num_nodes, dtype=torch.bool),
            "val": torch.zeros(num_nodes, dtype=torch.bool),
            "test": torch.zeros(num_nodes, dtype=torch.bool),
        }

        masks["train"][shuffled_indices[:num_train]] = True
        masks["val"][shuffled_indices[num_train : num_train + num_val]] = True
        masks["test"][shuffled_indices[num_train + num_val :]] = True

        # Update data object with new attributes
        data.train_mask = masks["train"].to(self.device)
        data.val_mask = masks["val"].to(self.device)
        data.test_mask = masks["test"].to(self.device)
        data.edge_index = edge_index.to(self.device)

        return data

    def train_epoch(
        self,
        model: GAT,
        data,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
    ) -> float:
        """Train for one epoch"""
        model.train()
        optimizer.zero_grad()

        # Forward pass
        output = model(data.x, data.edge_index)[
            0
        ]  # Get just the output, not attention weights
        loss = criterion(output[data.train_mask], data.y[data.train_mask])

        # Backward pass
        loss.backward()
        optimizer.step()

        return loss.item()

    def train(self, model: GAT, data) -> float:
        """Train the model"""
        # Prepare data
        data = self.prepare_data(data)
        model = model.to(self.device)

        # Setup optimizer and loss
        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.config.TRAIN_LEARNING_RATE
        )
        criterion = torch.nn.CrossEntropyLoss()

        best_val_acc = 0
        early_stopping_counter = 0

        # Training loop
        for epoch in range(self.config.TRAIN_EPOCHS):
            # Train for one epoch
            loss = self.train_epoch(model, data, optimizer, criterion)

            if epoch % 10 == 0:  # Evaluate every 10 epochs
                # Get metrics
                train_metrics = self.metrics.evaluate(model, data, data.train_mask)
                val_metrics = self.metrics.evaluate(model, data, data.val_mask)

                # Log metrics
                metrics = {f"train_{k}": v for k, v in train_metrics.items()}
                metrics.update({f"val_{k}": v for k, v in val_metrics.items()})
                metrics["train_loss"] = loss

                mlflow.log_metrics(metrics, step=epoch)

                # Early stopping check
                if val_metrics["accuracy"] > best_val_acc:
                    best_val_acc = val_metrics["accuracy"]
                    early_stopping_counter = 0
                    mlflow.pytorch.log_model(model, "model")
                else:
                    early_stopping_counter += 1

                if early_stopping_counter > self.config.TRAIN_EARLY_STOPPING_PATIENCE:
                    self.logger.info(
                        f"Early stopping triggered. Best validation accuracy: {best_val_acc}"
                    )
                    break

                # Log progress
                self.logger.info(
                    f"Epoch {epoch:03d}, Loss: {loss:.4f}, "
                    f"Train Acc: {train_metrics['accuracy']:.4f}, "
                    f"Val Acc: {val_metrics['accuracy']:.4f}"
                )

        return best_val_acc
