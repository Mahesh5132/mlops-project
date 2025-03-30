import mlflow

# Run ID (from your artifact path)
RUN_ID = "5ed6b58f2d90480da30b030ec357eff5"

# Define model artifact path
MODEL_PATH = f"mlflow-artifacts:/0/{RUN_ID}/artifacts/model"

# Download model to local directory
local_model_path = mlflow.artifacts.download_artifacts(MODEL_PATH)

print(f"Model downloaded to: {local_model_path}")
