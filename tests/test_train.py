import mlflow.pyfunc
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import os
from urllib.parse import urlparse

'''
def get_latest_model_path(select_index=0):
    """Dynamically fetch the latest MLflow model path."""
    mlflow.set_tracking_uri("http://localhost:5000")  # Set MLflow tracking URI
    client = mlflow.tracking.MlflowClient()

    # Get the latest run
    experiment_id = "0"  # Change if needed
    runs = client.search_runs(experiment_ids=[experiment_id], order_by=["start_time desc"], max_results=2)

    if not runs:
        raise ValueError("No MLflow runs found!")
    
    if select_index >= len(runs):
        raise IndexError(f"Requested model index {select_index} is out of range. Available models: {len(runs)}")
    
    selected_run = runs[select_index]
    artifact_uri = selected_run.info.artifact_uri

    # Convert MLflow artifact URI to local path
    if artifact_uri.startswith("mlflow-artifacts:/"):
        local_path = artifact_uri.replace("mlflow-artifacts:/", "C:/Users/Asus/Documents/MLOPS Projects/mlops-project/mlartifacts/")
    else:
        parsed_uri = urlparse(artifact_uri)
        if parsed_uri.scheme in ["file", ""]:
            local_path = os.path.abspath(parsed_uri.path)
        else:
            raise ValueError(f"Unsupported artifact URI scheme: {parsed_uri.scheme}")
    
    model_path = os.path.join(local_path, "artifacts", "model_placement_data_v2.csv")
   
  
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path does not exist: {model_path}") 

    return model_path


# Fetch the latest model path
MODEL_PATH = get_latest_model_path()
'''

MODEL_DIR = "C:/Users/Asus/Documents/MLOPS Projects/mlops-project/mlartifacts/0/5ed6b58f2d90480da30b030ec357eff5/artifacts/model_placement_data_v2.csv"

def test_model_loading():
    """Test if the MLflow model loads correctly."""
    try:
        model = mlflow.pyfunc.load_model(MODEL_DIR)
        assert model is not None, "Failed to load the model"
        print("✅ Model loaded successfully")
    except Exception as e:
        assert False, f"❌ Exception occurred while loading model: {e}"


def test_model_predictions():
    """Test if the model returns valid predictions."""
    try:
        model = mlflow.pyfunc.load_model(MODEL_DIR)

        test_data = pd.DataFrame({
            "CGPA": [3.5],  # Example value
            "Study_Hours": [5.0]  # Example value
        }).astype("float64")  # Ensure correct dtype

        predictions = model.predict(test_data)

        assert isinstance(predictions, (np.ndarray, list)), "Predictions should be an array or list"
     
        print("✅ Model predictions are valid")
    except Exception as e:
        assert False, f"❌ Exception occurred while testing model predictions: {e}"


if __name__ == "__main__":
    test_model_loading()
    test_model_predictions()
