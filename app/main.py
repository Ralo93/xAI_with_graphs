from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import numpy as np
from typing import List, Any
from app.model import *
import torch.nn.functional as F

# Consolidated configuration base class
class ConfigBase:
    class Action:
        ACTIVATION = F.relu
        DROPOUT = 0.5
        HIDDEN_DIM = 16
        NUM_LAYERS = 1
        INPUT_DIM = 128
        OUTPUT_DIM = 2
        AGG = 'mean'

    class Environment:
        # Environment Network Configuration
        INPUT_DIM = 1433
        OUTPUT_DIM = 7
        NUM_LAYERS = 3 #3 ALSO CHANGE IN MAIN.PY
        DROPOUT = 0.2
        HIDDEN_DIM = 128
        LAYER_NORM = False
        SKIP_CONNECTION = True
        AGG = 'sum'

    class Gumbel:
        TEMPERATURE = 0.5
        TAU = 0.01
        LEARN_TEMPERATURE = True

    class Training:
        LEARNING_RATE = 0.001
        WEIGHT_DECAY = 0
        EPOCHS = 200
        BATCH_SIZE = 32
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Specific configurations inheriting from base class
class Config:
    @staticmethod
    def Environment(input_dim, output_dim, num_layers):
        return type("Environment", (object,), {
            "INPUT_DIM": input_dim,
            "OUTPUT_DIM": output_dim,
            "NUM_LAYERS": num_layers,
            "DROPOUT": 0.5,
            "HIDDEN_DIM": 128,
            "LAYER_NORM": False,
            "SKIP_CONNECTION": True,
            "AGG": 'sum',
        })

configurations = {
    "3layer_cora": ConfigBase.Environment(1433, 7, 3),
    "5layer_cora": ConfigBase.Environment(1433, 7, 5),
    "3layer_re": ConfigBase.Environment(300, 18, 3),
    "5layer_re": ConfigBase.Environment(300, 18, 5),
    "10layer_re": ConfigBase.Environment(300, 18, 10),
}

# Load models dynamically
def load_model(name, config):
    if "cora" in name:
        return load_cognn_model_cora(config)
    elif "re" in name:
        return load_cognn_model_re(config)
    return None

models = {name: load_model(name, config) for name, config in configurations.items()}

# FastAPI application
app = FastAPI()

# Input and response schemas
class GraphInputData(BaseModel):
    node_features: List[List[float]]
    edge_index: List[List[int]]

class PredictionResponse(BaseModel):
    edge_index: List[List[int]]
    model_output: List[List[float]]
    class_probabilities: List[List[float]]
    attention_weights: List[Any]

class PredictionResponseCoGNN(PredictionResponse):
    edge_weights: List[List[float]]

# Prediction function template
def prediction_template(model, input_data, response_class):
    try:
        x = torch.tensor(input_data.node_features, dtype=torch.float32)
        edge_index = torch.tensor(input_data.edge_index, dtype=torch.long).t().contiguous()

        model.eval()
        with torch.no_grad():
            output, *extra_outputs = model(x, edge_index)
            probabilities = torch.softmax(output, dim=1).cpu().numpy()
            probabilities = np.nan_to_num(probabilities, nan=0.0, posinf=1.0, neginf=0.0)
            probabilities = [[float(p) for p in prob] for prob in probabilities]

            response = {
                "edge_index": edge_index.tolist(),
                "model_output": output.tolist(),
                "class_probabilities": probabilities,
            }

            if response_class == PredictionResponseCoGNN:
                edge_weights = [
                    ew.cpu().tolist() if isinstance(ew, torch.Tensor) else []
                    for ew in extra_outputs[0]
                ]
                response["edge_weights"] = edge_weights

            elif response_class == PredictionResponse:
                attention_weights = [aw.cpu().numpy().tolist() for aw in extra_outputs[0]]
                response["attention_weights"] = attention_weights

            return response

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# Endpoints for predictions
@app.post("/predict/{model_name}/", response_model=PredictionResponse)
async def predict(model_name: str, input_data: GraphInputData):
    if model_name not in models:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")

    model = models[model_name]
    response_class = PredictionResponse if "gat" in model_name else PredictionResponseCoGNN
    return prediction_template(model, input_data, response_class)