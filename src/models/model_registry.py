from pathlib import Path
from typing import Optional
import torch
from .gat import GAT, GATConfig


class ModelRegistry:
    DEFAULT_MODEL_PATH = Path("models/artifacts/model.pth")

    @staticmethod
    def save_model(model: GAT, path: Path) -> None:
        """Save model to specified path"""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), path)

    @staticmethod
    def load_model(config: GATConfig, model_path: Optional[Path] = None) -> GAT:
        """
        Load model with specified configuration

        Args:
            config: Model configuration
            model_path: Optional path to saved model weights

        Returns:
            GAT: Loaded model in evaluation mode
        """
        model = GAT(config)
        model_path = model_path or ModelRegistry.DEFAULT_MODEL_PATH

        try:
            if model_path.exists():
                state_dict = torch.load(model_path, map_location="cpu")
                if isinstance(state_dict, GAT):
                    # Handle case where entire model was saved
                    model = state_dict
                else:
                    # Handle case where only state dict was saved
                    model.load_state_dict(state_dict)
            else:
                print(f"Warning: Model file not found at {model_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

        return model.eval()

    @staticmethod
    def get_raw_model(config: GATConfig) -> GAT:
        """Create a new model instance without loading weights"""
        return GAT(config)
