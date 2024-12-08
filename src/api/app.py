from fastapi import FastAPI
from .routes import prediction
from ..config.settings import get_settings
from ..models.model_registry import ModelRegistry
from ..models.gat import GATConfig


def create_app() -> FastAPI:
    """Create and configure FastAPI application"""
    app = FastAPI(title="Graph Neural Network API")

    # Initialize settings and model
    settings = get_settings()
    config = GATConfig(
        in_channels=1433,  # Cora features
        hidden_channels=5,
        out_channels=7,  # Cora classes
        num_heads=5,
        dropout=0.3,
    )

    # Load model at startup
    settings.MODEL = ModelRegistry.load_model(
        config=config, model_path=settings.MODEL_PATH
    )

    # Register routes
    app.include_router(prediction.router, prefix="/api/v1")

    return app


# Create app instance for ASGI servers
app = create_app()
