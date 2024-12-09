# src/mlflow_main.py
import yaml
from pathlib import Path
import mlflow
import torch
from typing import Dict, Any

from src.data.cora import CoraDataset
from src.models.gat import GAT, GATConfig
from src.models.cognn import CoGNN, CoGNNConfig
from src.training.gat_trainer import GATTrainer
from src.training.cognn_trainer import CoGNNTrainer, CoGNNTrainingConfig
from src.config.settings import settings
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)


def load_config() -> Dict[str, Any]:
    """Load MLflow configuration from YAML file"""
    config_path = Path(__file__).parent / "config" / "mlflow_config.yml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def train_gat(data: CoraDataset, config: Dict[str, Any]):
    """Train GAT model"""
    # Create model config
    model_config = GATConfig(
        in_channels=1433,  # Cora features dimension
        hidden_channels=config["model"]["hidden_channels"],
        out_channels=config["out_dimenstions"],
        num_heads=config["model"]["num_heads"],
        dropout=config["model"]["dropout"],
    )

    # Create model
    model = GAT(model_config)
    logger.info(f"Created GAT model: {model}")

    # Initialize trainer
    trainer = GATTrainer(settings)

    # Train model
    best_val_acc = trainer.train(model, data)
    logger.info(f"GAT Training completed. Best validation accuracy: {best_val_acc:.4f}")

    return best_val_acc


def train_cognn(data: CoraDataset, config: Dict[str, Any]):
    """Train CoGNN model"""
    # Create model
    model = CoGNN(CoGNNConfig())
    logger.info(f"Created CoGNN model: {model}")

    # Initialize trainer
    training_config = CoGNNTrainingConfig(
        LEARNING_RATE=config["optimizer"]["lr"],
        EPOCHS=config["training"]["epochs"],
        DEVICE=torch.device(config["training"]["device"]),
    )
    trainer = CoGNNTrainer(training_config)

    # Train model
    best_val_acc = trainer.train(model, data)
    logger.info(
        f"CoGNN Training completed. Best validation accuracy: {best_val_acc:.4f}"
    )

    return best_val_acc


def main():
    # Load configuration
    config = load_config()

    # Initialize MLflow
    mlflow.set_tracking_uri(
        config.get("mlflow", {}).get("tracking_uri", "file:./mlruns")
    )
    mlflow.set_experiment(config["experiment_name"])

    # Load dataset
    dataset = CoraDataset("data/raw/cora.npz")
    logger.info(f"Loaded dataset: {dataset}")

    # Train models
    models = {"gat": train_gat, "cognn": train_cognn}

    model_type = config.get("model_type", "gat")
    if model_type not in models:
        raise ValueError(
            f"Unknown model type: {model_type}. Available types: {list(models.keys())}"
        )

    # Run training
    with mlflow.start_run():
        # Log general parameters
        mlflow.log_params(
            {
                "model_type": model_type,
                "dataset": "Cora",
                "learning_rate": config["optimizer"]["lr"],
                "epochs": config["training"]["epochs"],
            }
        )

        # Train selected model
        train_fn = models[model_type]
        best_val_acc = train_fn(dataset, config)

        # Log final metrics
        mlflow.log_metric("best_validation_accuracy", best_val_acc)


if __name__ == "__main__":
    main()
