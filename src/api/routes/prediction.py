from fastapi import APIRouter, HTTPException
import torch
import numpy as np
from typing import Any
from ..schemas.graph import GraphInputData, PredictionResponse
from ...models.model_registry import ModelRegistry
from ...config.settings import get_settings

router = APIRouter()


@router.post("/predict/", response_model=PredictionResponse)
async def predict(input_data: GraphInputData) -> PredictionResponse:
    try:
        # Convert input to PyTorch tensors
        x = torch.tensor(input_data.node_features, dtype=torch.float32)
        edge_index = (
            torch.tensor(input_data.edge_index, dtype=torch.long).t().contiguous()
        )

        # Get model from registry
        settings = get_settings()
        model = settings.MODEL
        model.eval()

        with torch.no_grad():
            # Get predictions and attention weights
            output, attention_weights = model(x, edge_index)

            # Compute probabilities
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
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
