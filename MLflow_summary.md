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
    - [Logging: Parameters, Metrics, Artifacts and Tags](#logging-parameters-metrics-artifacts-and-tags)
    - [Multiple Experiments and Runs](#multiple-experiments-and-runs)
    - [Autologging](#autologging)

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

# Infer the model signature: model input & output schemas
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
conda activate mlflow

# Serve web UI, call it from folder with ./mlruns in it
# Go to the folder where the experiment/runs are, e.g., we should see the mlruns/ folder
cd ...
mlflow ui
# Open http://127.0.0.1:5000 in browser
# The experiments and runs saved in the local mlruns folder are loaded
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

### Logging: Parameters, Metrics, Artifacts and Tags

```python
## -- Parameters

# Hyperparameters passed as key-value pairs
mlflow.log_param(key: str, value: Any) # single hyperparam -> Returns logged param!
mlflow.log_params(params: Dict[str, Any]) # multiple hyperparams -> No return
mlflow.log_params(params={"alpha": alpha, "l1_ratio": l1_ratio})

## -- Metrics

# Metrics passed as key-value pairs: RMSE, etc.
mlflow.log_metric(key: str, value: float, step: Optional[int] = None) # single -> No return
mlflow.log_metrics(metrics: Dict[str, float], step: Optional[int] = None) # multiple -> No return
mlflow.log_metrics(metrics={"mae": mae, "r2": r2})

## -- Artifacts

# Log artifacts: datasets, etc.
# We can also log models, but it's better to use mlflow.<framework>.log_model for that
# We pass the local_path where the artifact is
# and it will be stored in the mlruns folder, in the default path for the artifacts,
# unless we specify a different artifact_path
mlflow.log_artifact(local_path: str, artifact_path: Optional[str] = None) # single -> No return
# The multiple case takes a directory, and all the files within it are stored
# Use-cases: Computer Vision, folder with images; train/test splits
mlflow.log_artifacts(local_dir: str, artifact_path: Optional[str] = None) # multiple

# Example
# local dataset: org & train/test split
data = pd.read_csv("../data/red-wine-quality.csv")
Path("./data/").mkdir(parents=True, exist_ok=True)
data.to_csv("data/red-wine-quality.csv", index=False)
train, test = train_test_split(data)
train.to_csv("data/train.csv")
test.to_csv("data/test.csv")
mlflow.log_artifacts("data/")

# Get the absolute URI of an artifact
# If we input the artifact_path, the URI of the specific artifact is returned,
# else, the URI of the current artifact directory is returned.
artifacts_uri = mlflow.get_artifact_uri(artifact_path: Optional[str] = None)

artifacts_uri = mlflow.get_artifact_uri() # file://.../exp_xxx/run_yyy/artifacts
artifacts_uri = mlflow.get_artifact_uri(artifact_path="data/train.csv") # file://.../exp_xxx/run_yyy/artifacts/data/train.csv

## -- Tags
# Tags are used to group runs; mlflow creates also some internal tags automatically
# Tags are assigned to a run, so they can be set between start & end
mlflow.set_tag(key: str, value: Any) # single -> No return
mlflow.set_tags(tags: Dict[str, Any]) # multiple -> No return
mlflow.set_tags(tags={"version": "1.0", "environment": "dev"})
```

### Multiple Experiments and Runs

We can log multiple experiments and runs in the same file, we just need to control their name for that.
Cases in which that is interesting:

- *Incremental training*, i.e., we train until a given point and then we decide to continue doing it.
- *Model checkpoints*.
- *Hyperparameter tuning*: one run for eahc parameter set.
- *Cross-validation*: one run for each fold.
- *Feature engineering*: one run for each set of transformations.
- We can launch several experiments in a process; this makes sense when we are trying different models.

```python
# -- Experiment 1
exp = mlflow.set_experiment(experiment_name="experiment_1")
# Run 1
mlflow.start_run(run_name="run_1.1")
# ... do anthing
mlflow.end_run()
# Run 2
mlflow.start_run(run_name="run_1.2")
# ... do anything
mlflow.end_run()

# -- Experiment 2
exp = mlflow.set_experiment(experiment_name="experiment_2")
# Run 1
mlflow.start_run(run_name="run_1.1")
# ... do anthing
mlflow.end_run()
```

### Autologging

MLflow allows automatically logging parameters and metrics, without the need to specifying them explicitly. We just need to place `mlflow.autolog()` just before the model definition and training; then, all the model parameters and metrics are logged.

```python
# Generic autolog: the model library is detected and its logs are carried out
mlflow.autolog(log_models: boot = True, # log model or not
               log_input_examples: bool = False, # log input examples or not
               log_model_signatures: bool = True, # signatures: schema of inputs and outputs
               log_datasets: bool = True,
               disable: bool = False, # disable all automatic logging
               exclusive: bool = False, # if True, autologged content not logged to user-created runs
               disable_for_unsupported_versions: bool = False, # 
               silent: bool = False) # supress all event logs and warnings

# Library-specific, i.e., we explicitly specify the librar:
# sklearn, keras, xgboost, pytorch, spark, gluon, statsmodels, ...
# Same parameters as mlflow.autolog) + 5 additonal
mlflow.<framework>.autolog(...,
                           max_tuning_runs: int = 5, # max num of child MLflow runs for hyperparam search
                           log_post_training_metrics: bool = True, # metrics depend on model type
                           serialization_format: str = 'cloudpickle', # each library has its own set
                           registered_model_name: Optional[str] = None, # to serialize the model
                           pos_label: Optional[str] = None) # positive class in binary classification
mlflow.sklearn.autolog(...)

# Now we define and train the model
# ...
```