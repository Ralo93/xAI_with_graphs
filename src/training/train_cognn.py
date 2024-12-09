import mlflow
from pathlib import Path
from ..models.cognn import CoGNN, CoGNNConfig
from ..data.cora import CoraDataset
from .cognn_trainer import CoGNNTrainer, CoGNNTrainingConfig
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    # Load dataset
    dataset = CoraDataset("data/raw/cora.npz")
    logger.info(f"Loaded dataset: {dataset}")

    # Initialize model
    model = CoGNN(CoGNNConfig())
    logger.info(f"Initialized model: {model}")

    # Initialize trainer
    trainer = CoGNNTrainer(CoGNNTrainingConfig())

    # Setup MLflow
    mlflow.set_experiment("cognn-cora")

    # Train model
    with mlflow.start_run():
        # Log parameters
        mlflow.log_params(
            {
                "model_type": "CoGNN",
                "dataset": "Cora",
                "learning_rate": trainer.config.LEARNING_RATE,
                "weight_decay": trainer.config.WEIGHT_DECAY,
                "epochs": trainer.config.EPOCHS,
            }
        )

        best_val_acc = trainer.train(model, dataset)
        logger.info(f"Training completed. Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
