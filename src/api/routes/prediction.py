from fastapi import APIRouter, HTTPException
import torch
import numpy as np
from ..schemas.graph import GraphInputData, PredictionResponse
from ...models.model_registry import ModelRegistry
from ...models.gat import GATConfig
from ...config.settings import settings

router = APIRouter()


@router.post("/predict/", response_model=PredictionResponse)
async def predict(input_data: GraphInputData) -> PredictionResponse:
    try:
        # Convert input to PyTorch tensors
        x = torch.tensor(input_data.node_features, dtype=torch.float32)
        edge_index = (
            torch.tensor(input_data.edge_index, dtype=torch.long).t().contiguous()
        )
        print(f"Input tensor shapes: x={x.shape}, edge_index={edge_index.shape}")

        # Load model from registry
        config = GATConfig(
            in_channels=x.size(1),
            hidden_channels=settings.MODEL_HIDDEN_CHANNELS,
            out_channels=settings.MODEL_OUT_CHANNELS,
            num_heads=4
        )

        model = ModelRegistry.load_model(
            config=config,
            model_path=settings.MODEL_PATH,
            model_type=settings.MODEL_TYPE,
        )

        model.eval()
        with torch.no_grad():
            output, attention_weights = model(x, edge_index)
            probabilities = torch.softmax(output, dim=1)
            probabilities = np.nan_to_num(
                probabilities.cpu().numpy(), nan=0.0, posinf=1.0, neginf=0.0
            )

            return PredictionResponse(
                edge_index=edge_index.tolist(),
                model_output=output.tolist(),
                class_probabilities=probabilities.tolist(),
                attention_weights=[
                    aw[1].cpu().numpy().tolist() for aw in attention_weights
                ],
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
