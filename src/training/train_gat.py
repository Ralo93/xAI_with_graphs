from src.data.cora import CoraDataset
from src.models.gat import GAT, GATConfig
from src.training.gat_trainer import GATTrainer
from src.config.settings import settings


def main():
    # Load dataset
    dataset = CoraDataset("data/raw/cora.npz")
    print(f"Loaded dataset: {dataset}")

    # Create model config
    config = GATConfig(
        in_channels=1433,  # Cora features dimension
        hidden_channels=5,  # Hidden layer size
        out_channels=7,  # Number of classes in Cora
        num_heads=5,
        dropout=0.3,
    )

    # Create model
    model = GAT(config)
    print(f"Created model: {model}")

    # Initialize trainer
    trainer = GATTrainer(settings)

    # Train model
    best_val_acc = trainer.train(model, dataset)
    print(f"Training completed. Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
