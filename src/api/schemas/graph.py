from typing import List, Any
from pydantic import BaseModel


class GraphInputData(BaseModel):
    """Structured input for graph neural network prediction"""

    node_features: List[List[float]]  # 2D list of node features
    edge_index: List[List[int]]  # Edge connections


class PredictionResponse(BaseModel):
    """Structured prediction response"""

    edge_index: List[List[int]]
    model_output: List[List[float]]
    class_probabilities: List[List[float]]
    attention_weights: List[Any]
