from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import numpy as np
from typing import List
from app.model import *
from typing import List, Any
from pydantic import BaseModel


def load_model_in_main():
    try:
        model = load_model(
        in_channels=1433,      # Adjust to your node feature dimension
        hidden_channels=5, # Adjust to your model's hidden layer size
        out_channels=7,     # Number of output classes
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
    edge_index: List[List[int]]
    model_output: List[List[float]]
    class_probabilities: List[List[float]]  # Probabilities for each node
    attention_weights: List[Any]  # Attention weights from each layer


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
            # Get model prediction and attention weights
            output, attention_weights = model(x, edge_index)

            # Compute class probabilities
            probabilities = torch.softmax(output, dim=1)
            probabilities = probabilities.cpu().numpy()

            # Handle NaN or infinite values
            probabilities = np.nan_to_num(probabilities, nan=0.0, posinf=1.0, neginf=0.0)

            # Convert to a list of lists with Python floats
            probabilities = [[float(p) for p in prob] for prob in probabilities]
            
            # Convert attention weights to a serializable format (e.g., lists)
            attention_weights = [aw[1].cpu().numpy().tolist() for aw in attention_weights]

            # Debugging: Print shapes
            print("Attention Weights Debugging:")
            for i, aw in enumerate(attention_weights):
                print(f"Layer {i+1}: {len(aw)} edges, {len(aw[0]) if len(aw) > 0 else 0} heads")

            return {
                "edge_index": edge_index.tolist(),
                "model_output": output.tolist(),
                "class_probabilities": probabilities,
                "attention_weights": attention_weights,
            }
        
    except Exception as e:
        # Print full traceback for debugging
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
