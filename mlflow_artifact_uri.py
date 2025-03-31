from mlflow.tracking import MlflowClient
client = MlflowClient()
runs = client.search_runs(experiment_ids=["0"], order_by=["start_time desc"])
for run in runs:
    print(run.info.artifact_uri)
