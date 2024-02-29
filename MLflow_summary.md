# A Summary of the Most Important MLflow commands

This file contains a summary of the most important commands from [MLflow](https://mlflow.org/).

See the source guide in [`README.md`](./README.md).

Table of contents:
- [A Summary of the Most Important MLflow commands](#a-summary-of-the-most-important-mlflow-commands)
  - [Tracking: Basic Example](#tracking-basic-example)
  - [MLflow Server and UI](#mlflow-server-and-ui)
  - [Tracking: Logging](#tracking-logging)
    - [Experiments: Parameters](#experiments-parameters)
    - [Runs: Handling](#runs-handling)

## Tracking: Basic Example

```python
# Imports
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

# ...
# Fit model
# It is recommended to fit and evaluate the model outside
# of the `with` context in which the run is logged:
# in case something goes wrong, no run is created
lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
lr.fit(train_x, train_y)
# Predict and evaluate
predicted_qualities = lr.predict(test_x)
(rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

mlflow.set_tracking_uri("") # -> ./mlruns
# empty string: data saved automatically in ./mlruns
# local folder name: "./my_folder"
# file path: "file:/Users/username/path/to/file" (no C:)
# URL:
#     (local) "http://localhost:5000"
#     (remote) "https://my-tracking-server:5000"
# databricks workspace: "databricks://<profileName>"

# Create experiment, if not existent, else set it
exp = mlflow.set_experiment(experiment_name="experment_1")

# Infer the model signature
signature = infer_signature(train_x, lr.predict(train_x))

# Log run in with context
with mlflow.start_run(experiment_id=exp.experiment_id):    
    # Log: parameters, metrics, model itself
    mlflow.log_param("alpha", alpha)
    mlflow.log_param("l1_ratio", l1_ratio)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("mae", mae)
    model_info = mlflow.sklearn.log_model(
        sk_model=lr,
        artifact_path="wine_model", # dir name in the artifacts to dump model
        signature=signature,
        input_example=train_x[:2],
        # If registered_model_name is given, the model is registered!
        #registered_model_name=f"elasticnet-{alpha}-{l1_ratio}",
    )

# Load model and use it
loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
predictions = loaded_model.predict(test_x)
```

## MLflow Server and UI

```bash
# Serve web UI, call it from folder with ./mlruns in it
mlflow ui
# Open http://127.0.0.1:5000 in browser
# Experiments tab: runs and their params + metrics
# Models tab: model registry, if any

# Create a server: Then, we can use the URI in set_tracking_uri()
mlflow server --host 127.0.0.1 --port 8080
# URI: http://127.0.0.1:8080, http://localhost:8080
# To open the UI go to that URI with the browser

# To start the UI pointing to another tracking folder than mlruns
mlflow ui --backend-store-uri 'my_tracking'
```

## Tracking: Logging

### Experiments: Parameters

```python
from pathlib import Path

# Create new experiment
# - name: unique name
# - artifact_location: location to store run artifacts, default: artifacts
# - tags: optional dictionary of string keys and values to set tags
# Return: id
exp_id = mlflow.create_experiment(
    name="exp_create_exp_artifact",
    tags={"version": "v1", "priority": "p1"},
    artifact_location=Path.cwd().joinpath("myartifacts").as_uri() # must be a URI: file://...
)

exp = mlflow.get_experiment(exp_id)
print("Name: {}".format(exp.name)) # exp_create_exp_artifact
print("Experiment_id: {}".format(exp.experiment_id)) # 473668474626843335
print("Artifact Location: {}".format(exp.artifact_location)) # file:///C:/Users/.../mlflow_guide/examples/02_logging/myartifacts
print("Tags: {}".format(exp.tags)) # {'priority': 'p1', 'version': 'v1'}
print("Lifecycle_stage: {}".format(exp.lifecycle_stage)) # active
print("Creation timestamp: {}".format(exp.creation_time)) # 1709202652141

# Set existent experiment; not existent, it is created
# - name
# - experiment_id
# Return: experiment object itself, not the id as in create_experiment!
exp = mlflow.set_experiment(
    name="exp_create_exp_artifact"
)
```

### Runs: Handling

We can start runs outside from `with` contexts.

```python
# Start a run
# - run_id: optional; we can set it to overwrite runs, for instance
# - experiment_id: optional
# - run_name: optional, if run_id not specified
# - nested: to create a run within a run, set it to True
# - tags
# - description
# Returns: mlflow.ActiveRun context manager that can be used in `with` block
active_run = mlflow.start_run()

# If we don't call start_run() inside a with, we need to stop it
# - status = "FINISHED" (default), "SCHEDULED", "FAILED", "KILLED"
mlflow.end_run()

# Get current active run
# Returns ActiveRun context manager
active_run = mlflow.active_run()

# Get the last run which was active, called after end_run()
# Returns Run object
mlflow.end_run()
run = mlflow.last_active_run()
print("Active run id is {}".format(run.info.run_id)) # 02ae930f5f2348c6bc3b411bb7de297a
print("Active run name is {}".format(run.info.run_name)) # traveling-tern-43
```