import yaml
from pathlib import Path
import mlflow
from src.data.cora import CoraDataset
from src.models.gat import GAT, GATConfig
from src.training.trainer import GATTrainer
from src.config.settings import settings


def load_config():
    """Load MLflow configuration from YAML file"""
    config_path = Path(__file__).parent / "config" / "mlflow_config.yml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    # Load configuration
    config = load_config()

    # Initialize MLflow
    mlflow.set_tracking_uri(
        config.get("mlflow", {}).get("tracking_uri", "file:./mlruns")
    )
    mlflow.set_experiment(config["experiment_name"])

    # Optional: print configuration for debugging
    print("Loaded configuration:", config)

    # Load dataset
    dataset = CoraDataset("data/raw/cora.npz")
    print(f"Loaded dataset: {dataset}")

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
    print(f"Created model: {model}")

    # Initialize trainer with config
    trainer = GATTrainer(settings)

    # Train model
    best_val_acc = trainer.train(model, dataset)
    print(f"Training completed. Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
