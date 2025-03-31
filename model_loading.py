import pickle

MODEL_PATH = r"C:/Users/Asus/Documents/MLOPS Projects/mlops-project/mlartifacts/0/5ed6b58f2d90480da30b030ec357eff5/artifacts/model_placement_data_v2.csv/model.pkl"

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

print(type(model))  # Should print the type of the loaded model

import mlflow.pyfunc

MODEL_DIR = "C:/Users/Asus/Documents/MLOPS Projects/mlops-project/mlartifacts/0/5ed6b58f2d90480da30b030ec357eff5/artifacts/model_placement_data_v2.csv"
model = mlflow.pyfunc.load_model(MODEL_DIR)
print(model.metadata)

