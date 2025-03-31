import yaml
import mlflow

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
