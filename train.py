import os
import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from mlflow.models import infer_signature

# Set valid MLflow tracking URI
mlflow.set_tracking_uri("http://localhost:5000")  # If MLflow server is running

def train_and_log_model(input_dir):
    # Loop through all CSV files in the processed directory
    for file_name in os.listdir(input_dir):
        if file_name.endswith(".csv"):  # Process only CSV files
            input_path = os.path.join(input_dir, file_name)
            
            # Load data
            df = pd.read_csv(input_path)
            if "Placement_Package" not in df.columns:
                print(f"Skipping {file_name}: No Placement_Package column found.")
                continue
            
            X = df.drop("Placement_Package", axis=1)
            y = df["Placement_Package"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            
            # Train model
            model = RandomForestRegressor(n_estimators=100)
            model.fit(X_train, y_train)

            # Prepare input example and signature
            input_example = X_test.iloc[:1].to_dict(orient="records")
            signature = infer_signature(X_test, model.predict(X_test))
            
            # Log experiment
            with mlflow.start_run(run_name=file_name):
                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                mlflow.log_metric("mse", mse)
                mlflow.sklearn.log_model(model, f"model_{file_name}",
                                         input_example=input_example, 
                                         signature=signature)
                print(f"Trained and logged model for {file_name} with MSE: {mse}")

if __name__ == "__main__":
    processed_dir = "data/processed"
    train_and_log_model(processed_dir)
