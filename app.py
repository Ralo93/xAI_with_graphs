from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.pyfunc
import pandas as pd

# Load the model from MLflow
MODEL_URI = "file:C:/Users/rapha/repositories/final_project/heterophilous-graphs/mlruns/10/ec80f1fe720946fe9f11126ec5831338/artifacts/model"  # Adjust the model URI as needed
try:
    model = mlflow.pyfunc.load_model(MODEL_URI)
except Exception as e:
    raise RuntimeError(f"Failed to load the model: {e}")

# Define the app
app = FastAPI()

# Define the input schema
class InputData(BaseModel):
    features: list

@app.post("/predict/")
async def predict(input_data: InputData):
    try:
        # Convert input to DataFrame
        data = pd.DataFrame([input_data.features])
        
        # Make prediction
        prediction = model.predict(data)
        
        # Return prediction as response
        return {"prediction": prediction.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
