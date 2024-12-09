import torch
import numpy as np
from typing import Dict
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
)


class MetricsCalculator:
    """Calculator for various model evaluation metrics"""

    @staticmethod
    def evaluate(
        model: torch.nn.Module, data: any, mask: torch.Tensor
    ) -> Dict[str, float]:
        """
        Evaluate the model using various metrics

        Args:
            model: The GNN model
            data: The graph data object
            mask: Boolean mask indicating which nodes to evaluate

        Returns:
            Dictionary of metric names and values
        """
        model.eval()
        with torch.no_grad():
            # Get model outputs
            outputs = model(data.x, data.edge_index)
            if isinstance(outputs, tuple):
                outputs = outputs[
                    0
                ]  # In case model returns (outputs, attention_weights)

            outputs = outputs[mask]
            targets = data.y[mask]

            # For multi-class classification, use softmax
            if outputs.size(-1) > 1:
                probs = torch.softmax(outputs, dim=-1)
                preds = outputs.argmax(dim=-1)
            else:
                # For binary classification
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).long()

            # Convert to numpy for metric calculation
            targets_np = targets.cpu().numpy()
            preds_np = preds.cpu().numpy()
            probs_np = probs.cpu().numpy()

            # Calculate metrics
            metrics = {
                "accuracy": accuracy_score(targets_np, preds_np),
                "f1_macro": f1_score(
                    targets_np, preds_np, average="macro", zero_division=0
                ),
                "f1_weighted": f1_score(
                    targets_np, preds_np, average="weighted", zero_division=0
                ),
                "precision_macro": precision_score(
                    targets_np, preds_np, average="macro", zero_division=0
                ),
                "recall_macro": recall_score(
                    targets_np, preds_np, average="macro", zero_division=0
                ),
            }

            # Calculate AUC-ROC for multi-class
            try:
                if outputs.size(-1) > 1:  # Multi-class
                    metrics["auc"] = roc_auc_score(
                        targets_np, probs_np, multi_class="ovr"
                    )
                else:  # Binary
                    metrics["auc"] = roc_auc_score(targets_np, probs_np.squeeze())
            except ValueError:
                # Handle cases where AUC is undefined (e.g., single class in fold)
                metrics["auc"] = float("nan")

            return metrics
