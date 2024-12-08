from dataclasses import dataclass
from typing import Dict, Any, Optional
import torch
import mlflow
import logging
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops
from ..models.gat import GAT, GATConfig
from ..utils.metrics import MetricsCalculator
from ..utils.graph import GraphUtils


@dataclass
class TrainingConfig:
    """Training configuration parameters"""

    device: str
    epochs: int
    early_stopping_patience: int
    train_split: float
    val_split: float
    learning_rate: float


class GATTrainer:
    """Trainer class for Graph Attention Networks"""

    def __init__(self, config: TrainingConfig, logger: Optional[logging.Logger] = None):
        self.config = config
        self.device = torch.device(config.device)
        self.logger = logger or logging.getLogger(__name__)
        self.metrics = MetricsCalculator()
        self.graph_utils = GraphUtils()

    def prepare_data(self, data: Data) -> Data:
        """Prepare data for training"""
        # Add self-loops and make bidirectional
        edge_index = self.graph_utils.make_bidirectional(data.edge_index)
        edge_index = add_self_loops(edge_index=edge_index)[0]

        # Create train/val/test masks
        num_nodes = data.x.size(0)
        shuffled_indices = torch.randperm(num_nodes)

        num_train = int(self.config.train_split * num_nodes)
        num_val = int(self.config.val_split * num_nodes)

        masks = {
            "train": torch.zeros(num_nodes, dtype=torch.bool),
            "val": torch.zeros(num_nodes, dtype=torch.bool),
            "test": torch.zeros(num_nodes, dtype=torch.bool),
        }

        masks["train"][shuffled_indices[:num_train]] = True
        masks["val"][shuffled_indices[num_train : num_train + num_val]] = True
        masks["test"][shuffled_indices[num_train + num_val :]] = True

        return Data(
            x=data.x,
            edge_index=edge_index,
            y=data.y,
            train_mask=masks["train"],
            val_mask=masks["val"],
            test_mask=masks["test"],
        ).to(self.device)

    def train_epoch(
        self,
        model: GAT,
        data: Data,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
    ) -> float:
        """Train for one epoch"""
        model.train()
        optimizer.zero_grad()

        out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])

        loss.backward()
        optimizer.step()

        return loss.item()

    def train(self, model: GAT, data: Data, trial: Optional[Any] = None) -> float:
        """Train the model"""
        data = self.prepare_data(data)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
        criterion = torch.nn.CrossEntropyLoss()

        best_val_acc = 0
        early_stopping_counter = 0

        with mlflow.start_run():
            # Log model parameters
            model_params = {
                name: param.data.numpy().tolist()
                for name, param in model.named_parameters()
            }
            mlflow.log_params(model_params)

            for epoch in range(self.config.epochs):
                loss = self.train_epoch(model, data, optimizer, criterion)

                if epoch % 10 == 0:
                    metrics = self._evaluate_and_log(model, data, loss, epoch)

                    # Early stopping check
                    if metrics["val_accuracy"] > best_val_acc:
                        best_val_acc = metrics["val_accuracy"]
                        early_stopping_counter = 0
                        mlflow.pytorch.log_model(model, "model")
                    else:
                        early_stopping_counter += 1

                    if early_stopping_counter > self.config.early_stopping_patience:
                        self.logger.info(
                            f"Early stopping triggered. Best validation accuracy: {best_val_acc}"
                        )
                        break

            # Final evaluation
            test_metrics = self.metrics.evaluate(model, data, data.test_mask)
            mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

            return best_val_acc

    def _evaluate_and_log(
        self, model: GAT, data: Data, train_loss: float, epoch: int
    ) -> Dict[str, float]:
        """Evaluate model and log metrics"""
        train_metrics = self.metrics.evaluate(model, data, data.train_mask)
        val_metrics = self.metrics.evaluate(model, data, data.val_mask)

        metrics = {f"train_{k}": v for k, v in train_metrics.items()}
        metrics.update({f"val_{k}": v for k, v in val_metrics.items()})
        metrics["train_loss"] = train_loss

        mlflow.log_metrics(metrics, step=epoch)

        # Log progress
        self.logger.info(
            f"Epoch {epoch:03d}, Loss: {train_loss:.4f}, "
            f"Train Acc: {train_metrics['accuracy']:.4f}, "
            f"Val Acc: {val_metrics['accuracy']:.4f}"
        )

        return metrics
