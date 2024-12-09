from pathlib import Path
from functools import lru_cache
from typing import Optional
import torch
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings"""

    # API Settings
    API_PORT: int = 8080
    API_HOST: str = "0.0.0.0"
    DEBUG: bool = True
    API_PREFIX: str = "/api/v1"

    # Model Settings
    MODEL_PATH: Path = Path("models/artifacts/model.pth")
    MODEL_IN_CHANNELS: int = 1433
    MODEL_HIDDEN_CHANNELS: int = 5
    MODEL_OUT_CHANNELS: int = 7
    MODEL_NUM_HEADS: int = 5
    MODEL_DROPOUT: float = 0.3
    MODEL: Optional[torch.nn.Module] = None

    # MLflow Settings
    MLFLOW_TRACKING_URI: str = "http://localhost:5000"
    MLFLOW_EXPERIMENT_NAME: str = "gat-cora"
    MLFLOW_REGISTRY_URI: str = "models/artifacts"

    # Data Settings
    DATA_DIR: Path = Path("data")
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
    INTERIM_DATA_DIR: Path = DATA_DIR / "interim"

    # Training Settings
    TRAIN_BATCH_SIZE: int = 32
    TRAIN_EPOCHS: int = 1500
    TRAIN_LEARNING_RATE: float = 0.01
    TRAIN_EARLY_STOPPING_PATIENCE: int = 10
    TRAIN_SPLIT: float = 0.8
    VAL_SPLIT: float = 0.2
    USE_CUDA: bool = True

    # Logging Settings
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Model configuration using SettingsConfigDict
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=True
    )

    @property
    def device(self) -> torch.device:
        """Get the device to use for computation"""
        if self.USE_CUDA and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    @property
    def model_config_dict(self) -> dict:
        """Get model configuration"""
        return {
            "in_channels": self.MODEL_IN_CHANNELS,
            "hidden_channels": self.MODEL_HIDDEN_CHANNELS,
            "out_channels": self.MODEL_OUT_CHANNELS,
            "num_heads": self.MODEL_NUM_HEADS,
            "dropout": self.MODEL_DROPOUT,
        }


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


# Initialize settings instance
settings = get_settings()
