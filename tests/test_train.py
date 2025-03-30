import mlflow.pyfunc
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import os

def get_latest_model_path():
    """Dynamically fetch the latest MLflow model path."""
    mlflow.set_tracking_uri("http://localhost:5000")  # Set MLflow tracking URI
    client = mlflow.tracking.MlflowClient()

    # Get the latest run
    experiment_id = "0"  # Change if needed
    runs = client.search_runs(experiment_ids=[experiment_id], order_by=["start_time desc"], max_results=1)

    if not runs:
        raise ValueError("No MLflow runs found!")

    latest_run = runs[0]
    run_id = latest_run.info.run_id

    # Get the model artifact path dynamically
    artifact_uri = latest_run.info.artifact_uri  # Example: "mlartifacts:/0/abc123/artifacts"
    
    # Convert MLflow path to local system path
    model_path = mlflow.artifacts.download_artifacts(artifact_uri + "/model")  

    return model_path

# Fetch the latest model path
MODEL_PATH = get_latest_model_path()

def test_model_loading():
    """Test if the MLflow model loads correctly."""
    try:
        model = mlflow.pyfunc.load_model(MODEL_PATH)
        assert model is not None, "Failed to load the model"
        print("✅ Model loaded successfully")
    except Exception as e:
        assert False, f"❌ Exception occurred while loading model: {e}"


def test_model_predictions():
    """Test if the model returns valid predictions."""
    try:
        model = mlflow.pyfunc.load_model(MODEL_PATH)

        # Sample test data
        test_data = pd.DataFrame({"feature1": [1.5], "feature2": [3.2]})

        predictions = model.predict(test_data)

        assert isinstance(predictions, (np.ndarray, list)), "Predictions should be an array or list"
        assert len(predictions) == len(test_data), f"Expected {len(test_data)} predictions, but got {len(predictions)}"
        
        print("✅ Model predictions are valid")
    except Exception as e:
        assert False, f"❌ Exception occurred while testing model predictions: {e}"


def test_model_accuracy():
    """Test if the model achieves a reasonable Mean Squared Error (MSE)."""
    try:
        model = mlflow.pyfunc.load_model(MODEL_PATH)

        # Check if test files exist
        if not os.path.exists("data/processed/X_test.csv") or not os.path.exists("data/processed/y_test.csv"):
            raise FileNotFoundError("Test dataset not found. Ensure X_test.csv and y_test.csv exist.")

        X_test = pd.read_csv("data/processed/X_test.csv")
        y_test = pd.read_csv("data/processed/y_test.csv")

        y_test = y_test.values.flatten()  # Ensure it's a 1D array if needed
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)

        assert mse < 5, f"MSE is too high: {mse:.4f}"
        print(f"✅ Model accuracy test passed with MSE: {mse:.4f}")
    except Exception as e:
        assert False, f"❌ Exception occurred while evaluating model accuracy: {e}"


if __name__ == "__main__":
    test_model_loading()
    test_model_predictions()
    test_model_accuracy()
