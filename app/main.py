from fastapi import FastAPI, HTTPException
import mlflow
from pydantic import BaseModel
import torch
import numpy as np
import torch_geometric
from typing import List, Union
from app.model import *

import os

def load_model_in_main():
    try:
        model = load_model(
        in_channels=1433,      # Adjust to your node feature dimension
        hidden_channels=64, # Adjust to your model's hidden layer size
        out_channels=6,     # Number of output classes
        model_path=None     # Uses default path
        )
        return model
    except Exception as e:
        print(f"Failed to load the model: {e}")
        raise RuntimeError(f"Failed to load the model: {e}")

# Load model at startup
model = load_model_in_main()

print(model)

# Define the app
app = FastAPI()

class GraphInputData(BaseModel):
    """
    Structured input for graph neural network prediction
    Includes node features and edge connections
    """
    node_features: List[List[float]]  # 2D list of node features
    edge_index: List[List[int]]       # Edge connections


class PredictionResponse(BaseModel):
    """
    Structured prediction response
    """
    class_probabilities: List[List[float]]  # Probabilities for each node



@app.post("/predict/", response_model=PredictionResponse)
async def predict(input_data: GraphInputData):
    try:
        # Convert input to PyTorch tensors
        x = torch.tensor(input_data.node_features, dtype=torch.float32)
        edge_index = torch.tensor(input_data.edge_index, dtype=torch.long).t().contiguous()
        
        # Ensure model is in evaluation mode
        model.eval()
        
        # Disable gradient computation for inference
        with torch.no_grad():
            # Get model prediction
            output = model(x, edge_index)
 
            probabilities = torch.softmax(output, dim=1)
            # Validate and clean the probabilities
            probabilities = probabilities.cpu().numpy()
            probabilities = np.nan_to_num(probabilities, nan=0.0, posinf=1.0, neginf=0.0)

            #print(probabilities)
            # Convert to a list of lists with Python floats
            probabilities = [[float(p) for p in prob] for prob in probabilities]
            
            return {
                "class_probabilities": probabilities
            }
        
    except Exception as e:
        # Print full traceback
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))