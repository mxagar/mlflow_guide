# A Summary of the Most Important MLflow commands

This file contains a summary of the most important commands from [MLflow](https://mlflow.org/).

See the source guide in [`README.md`](./README.md).

Table of contents:
- [A Summary of the Most Important MLflow commands](#a-summary-of-the-most-important-mlflow-commands)
  - [Tracking: Basic Example](#tracking-basic-example)
  - [MLflow Server, UI and Storage](#mlflow-server-ui-and-storage)
  - [Tracking or Logging Component](#tracking-or-logging-component)
    - [Experiments: Parameters](#experiments-parameters)
    - [Runs: Handling](#runs-handling)
    - [Logging: Parameters, Metrics, Artifacts and Tags](#logging-parameters-metrics-artifacts-and-tags)
    - [Multiple Experiments and Runs](#multiple-experiments-and-runs)
    - [Autologging](#autologging)
  - [Model Component](#model-component)
    - [Signatures and Input Example](#signatures-and-input-example)
    - [API: Log, Save, Load](#api-log-save-load)
    - [Custom Libraries/Models](#custom-librariesmodels)
    - [(Simple) Evaluation](#simple-evaluation)
    - [Evaluation with Custom Metrics, Artifacts and Baseline Model](#evaluation-with-custom-metrics-artifacts-and-baseline-model)
  - [Model Registry Component](#model-registry-component)
    - [API Calls](#api-calls)
  - [Project Component](#project-component)
    - [Running Projects via CLI](#running-projects-via-cli)
    - [Running Projects via the Python API](#running-projects-via-the-python-api)
  - [MLflow Client](#mlflow-client)
  - [CLI Commands](#cli-commands)
  - [Integration with AWS SageMaker](#integration-with-aws-sagemaker)

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

## MLflow Server, UI and Storage

We can open the MLflow UI in several ways:

- Via `mlflow ui`: we open the UI by default reading the local `mlruns` folder.
- Via `mlflow server`: we start a server and the `mlflow` library communicates with that server via REST; when a server is launched, we define a URI &mdash; if we open the URI, we see the UI.

Even though for the user starting or not starting the server seems to have minimal effects on the operations (only the URI needs to be set), the underlying architecture is different:

- When no server is launched, `mlflow` is used as a library which creates/stores some files.
- When a server is launched, the `mlflow` library communicates to a server (REST) which creates/stores some files.

Additionally, we can store two types of things either locally or remotely:

- Backend store: parameters, metrics, tags, etc. These are stored either in files or in SQL DBs (locally or remotely): SQLite, PostgeSQL, MySQL, etc.
- Artifact store: artifacts, models, images, etc. These are stored either in local folders or in remote/cloud storage: Amazon S3, etc.

Here are some commands for openining the UI or launching a server with different storage configurations:

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

# MLflow locally with Tracking Server
# --backend-store-uri: We specify our backend store, here a SQLite DB
# --default-artifact-root: Directory where artifacts are stored, by default mlruns, here ./mlflow-artifacts 
# --host, --port: Where the server is running, and the port; here localhost:5000
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlflow-artifacts --host 127.0.0.1 --port 5000
# Then, we can browse http.://127.0.0.1:5000
# In the exeperimentes, the tracking URI is http.://127.0.0.1:5000

# Remote and Distributed: MLflow with remote Tracking Server and cloud/remote storages
mlflow server --backend-store-uri postgresql://user:password@postgres:5432/mlflowdb --default-artifact-root s3://bucket_name --host remote_host --no-serve-artifacts
```

## Tracking or Logging Component

### Experiments: Parameters

```python
from pathlib import Path

# ...

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
# ...

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

If we activate the autologging but would like to still log manually given things (e.g., models), we need to de-activate the autologging for those things in the `mlflow.autolog()` call.

```python
# ...

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

## Model Component

This component allows to save and serve models in a reproducible manner, i.e., not only the serialized model is stored, but also all the information related to its environment, methods used, traceability information, etc.

If we save a model locally using `mlflow.log_model()`, we'll get a local folder in the run `artifacts` with the following files:

```bash
conda.yaml            # conda environment
input_example.json    # few rows of the dataset that serve as input example
MLmodel               # YAML, most important file: model packagaing described and referenced here
model.pkl             # serialized model binary
python_env.yaml       # virtualenv 
requirements.txt      # dependencies for virtualenv
```

### Signatures and Input Example

The model signature describes the data input and output types and names of the model, i.e., the schemas. These signatures can be enforced at different levels, similarly as with Pydantic. We can create signatures manually (not recommended) or infer them automatically (recommended).

```python
from mlflow.models.signature import infer_signature

signature = infer_signature(X_test, predicted_qualities)

input_example = {
    "columns": np.array(X_test.columns),
    "data": np.array(X_test.values)
}

mlflow.sklearn.log_model(lr, "model", signature=signature, input_example=input_example)
```

### API: Log, Save, Load

These are the library calls to store standardized models or interact with them:

```python
# ...

# Model saved to a passed directory: only two flavors: sklearn and pyfunc
mlflow.save_model(
  sk_model, # model
  path, 
  conda_env, # path to a YAML or a dictionary
  code_paths, # list of local filesystems paths, i.e. code files used to create the model,
  mlflow_model, # flavor
  serialization_format,
  signature,
  input_example,
  pip_requirements, # path or list of requirements as strings; not necessary, these are inferred
  extra_pip_requirements, # we can leave MLflow to infer and then add some explicitly
  pyfunc_predict_fn, # name of the prediction function, e.g., 'predict_proba'
  metadata
)

# Model logged to a local/remote server, which stores it as configured
# The main difference is that the servers handles it in the model artifacts (locally or remotely)
# whereas save_model always stores the model locally.
# Same parameters as save_model, but some new/different ones
mlflow.log_model(
  artifact_path, # path for the artifact
  registered_model_name, # register the model
  await_registration_for
)

# Load both the logged/saved model
# If the model is registered (see model registry section), we can use the URI models:/<name>/<version>
mlflow.load_model(
  model_uri, # the model URI: /path/to/model, s3://buckect/path/to/model, models:/<name>/<version>, etc.
  dst_path # path to download the model to
)
```

### Custom Libraries/Models

We sometimes work with models based on libraries/framworks that are not supported by MLflow, e.g., when we have our own ML library. For those cases, we can log the model information using `mlflow.pyfunc.log_model()`.

In the following example code, we assume that MLflow does not support Scikit-Learn, so we are going to create a Python model with it, then log it and load it. Notes:

- We cannot use `mlflow.sklearn.log_param/metric()` functions, but instead, `mlflow.log_param/metric()`.
- We cannot use `mlflow.log_model()`, but instead `mlflow.pyfunc.log_model()`.
- Imports, `mlflow.start_run()`, `mlflow.end_run()`, etc. are ommitted, but the shown lines should be wrapped by them.
- The complete example is in [`06_custom_libraries/load_custom_model.py`](./examples/06_custom_libraries/load_custom_model.py).

```python
# ...

# Data artifacts
data = pd.read_csv("../data/red-wine-quality.csv")
train, test = train_test_split(data)
data_dir = 'data'
Path(data_dir).mkdir(parents=True, exist_ok=True)
data.to_csv(data_dir + '/dataset.csv')
train.to_csv(data_dir + '/dataset_train.csv')
test.to_csv(data_dir + '/dataset_test.csv')

# Model artifact: we serialize the model with joblib
model_dir = 'models'
Path(model_dir).mkdir(parents=True, exist_ok=True)
model_path = model_dir + "/model.pkl"
joblib.dump(lr, model_path)

# Artifacts' paths: model and data
# This dictionary is fetsched later by the mlflow context
artifacts = {
    "model": model_path,
    "data": data_dir
}

# We create a wrapper class, i.e.,
# a custom mlflow.pyfunc.PythonModel
#   https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#mlflow.pyfunc.PythonModel
# We need to define at least two functions:
# - load_context
# - predict
# We can also define further custom functions if we want
class ModelWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.model = joblib.load(context.artifacts["model"])

    def predict(self, context, model_input):
        return self.model.predict(model_input.values)

# Conda environment
conda_env = {
    "channels": ["conda-forge"],
    "dependencies": [
        f"python={sys.version}", # Python version
        "pip",
        {
            "pip": [
                f"mlflow=={mlflow.__version__}",
                f"scikit-learn=={sklearn.__version__}",
                f"cloudpickle=={cloudpickle.__version__}",
            ],
        },
    ],
    "name": "my_env",
}

# Log model with all the structures defined above
# We'll see all the artifacts in the UI: data, models, code, etc.
mlflow.pyfunc.log_model(
    artifact_path="custom_mlflow_pyfunc", # the path directory which will contain the model
    python_model=ModelWrapper(), # a mlflow.pyfunc.PythonModel, defined above
    artifacts=artifacts, # dictionary defined above
    code_path=[str(__file__)], # Code file(s), must be in local dir: "model_customization.py"
    conda_env=conda_env
)

# Usually, we would load the model in another file/session, not in the same run,
# however, here we do it in the same run.
# To load the model, we need to pass the model_uri, which can have many forms
#   https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#mlflow.pyfunc.load_model
# One option:
#   runs:/<mlflow_run_id>/run-relative/path/to/model, e.g., runs:/98dgxxx/custom_mlflow_pyfunc
# Usually, we'll get the run_id we want from the UI/DB, etc.; if it's the active run, we can fetch it
active_run = mlflow.active_run()
run_id = active_run.info.run_id
loaded_model = mlflow.pyfunc.load_model(model_uri=f"runs:/{run_id}/{model_artifact_path}")

# Predict
predicted_qualities = loaded_model.predict(test_x)

# Evaluate
(rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)
```

### (Simple) Evaluation

MLflow provides evaluation functinalities for MLflow packaged models (pyfunc flavor) via the call `mlflow.evaluate()`. We get evaluation metrics, plots and explanations (SHAP). Note: we need to manually `pip install shap`, if not done yet.

sThe evaluation metrics and artifacts (e.g., plots) are saved along with other metrics and artifacts.

```python
# ...

# Log model with all the structures defined above
# We'll see all the artifacts in the UI: data, models, code, etc.
model_artifact_path = "custom_mlflow_pyfunc"
mlflow.pyfunc.log_model(
    artifact_path=model_artifact_path, # the path directory which will contain the model
    python_model=ModelWrapper(), # a mlflow.pyfunc.PythonModel, defined above
    artifacts=artifacts, # dictionary defined above
    code_path=[str(__file__)], # Code file(s), must be in local dir: "model_customization.py"
    conda_env=conda_env
)

# Get model URI and evaluate
# NOTE: the default evaluator uses shap -> we need to manually pip install shap
model_artifact_uri = mlflow.get_artifact_uri(model_artifact_path)
mlflow.evaluate(
    model_artifact_uri, # model URI
    test, # test split
    targets="quality",
    model_type="regressor",
    evaluators=["default"] # if default, shap is used -> pip install shap
)
```

### Evaluation with Custom Metrics, Artifacts and Baseline Model

We can make more sophisticated evaluations by adding:

- Custom evaluation metrics, defined by own functions.
- Custom evaluation artifacts (e.g., plots), also defined by own functions.
- Baseline models against which metrics are compared, after setting thresholds.

A complete example is provided in [`07_evaluation/validation_threshold.py`](./examples/07_evaluation/validation_threshold.py). Here, the most important lines are summarized:

```python
from mlflow.models import make_metric
from sklearn.dummy import DummyRegressor
from mlflow.models import MetricThreshold

# ...

# Model artifact: we serialize the model with joblib
model_dir = 'models'
Path(model_dir).mkdir(parents=True, exist_ok=True)
model_path = model_dir + "/model.pkl"
joblib.dump(lr, model_path)

# Artifacts' paths: model and data
# This dictionary is fetsched later by the mlflow context
artifacts = {
    "model": model_path,
    "data": data_dir
}

# Save baseline model and an artifacts dictionary
baseline_model_path = model_dir + "/baseline_model.pkl"
joblib.dump(baseline_model, baseline_model_path)
baseline_artifacts = {
    "baseline_model": baseline_model_path
}

# We create a wrapper class, i.e.,
# a custom mlflow.pyfunc.PythonModel
#   https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#mlflow.pyfunc.PythonModel
# We need to define at least two functions:
# - load_context
# - predict
# We can also define further custom functions if we want
class ModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, artifacts_name):
        # We use the artifacts_name in order to handle both the baseline & the custom model
        self.artifacts_name = artifacts_name

    def load_context(self, context):
        self.model = joblib.load(context.artifacts[self.artifacts_name])

    def predict(self, context, model_input):
        return self.model.predict(model_input.values)

# Conda environment
conda_env = {
    "channels": ["conda-forge"],
    "dependencies": [
        f"python={sys.version}", # Python version
        "pip",
        {
            "pip": [
                f"mlflow=={mlflow.__version__}",
                f"scikit-learn=={sklearn.__version__}",
                f"cloudpickle=={cloudpickle.__version__}",
            ],
        },
    ],
    "name": "my_env",
}

# Log model with all the structures defined above
# We'll see all the artifacts in the UI: data, models, code, etc.
mlflow.pyfunc.log_model(
    artifact_path="custom_mlflow_pyfunc", # the path directory which will contain the model
    python_model=ModelWrapper("model"), # a mlflow.pyfunc.PythonModel, defined above
    artifacts=artifacts, # dictionary defined above
    code_path=[str(__file__)], # Code file(s), must be in local dir: "model_customization.py"
    conda_env=conda_env
)

# Baseline model
mlflow.pyfunc.log_model(
    artifact_path="baseline_mlflow_pyfunc", # the path directory which will contain the model
    python_model=ModelWrapper("baseline_model"), # a mlflow.pyfunc.PythonModel, defined above
    artifacts=baseline_artifacts, # dictionary defined above
    code_path=[str(__file__)], # Code file(s), must be in local dir: "model_customization.py"
    conda_env=conda_env
)

def squared_diff_plus_one(eval_df, _builtin_metrics):
    return np.sum(np.abs(eval_df["prediction"] - eval_df["target"] + 1) ** 2)

def sum_on_target_divided_by_two(_eval_df, builtin_metrics):
    return builtin_metrics["sum_on_target"] / 2

squared_diff_plus_one_metric = make_metric(
    eval_fn=squared_diff_plus_one,
    greater_is_better=False,
    name="squared diff plus one"
)

sum_on_target_divided_by_two_metric = make_metric(
    eval_fn=sum_on_target_divided_by_two,
    greater_is_better=True,
    name="sum on target divided by two"
)

def prediction_target_scatter(eval_df, _builtin_metrics, artifacts_dir):
    plt.scatter(eval_df["prediction"], eval_df["target"])
    plt.xlabel("Targets")
    plt.ylabel("Predictions")
    plt.title("Targets vs. Predictions")
    plot_path = os.path.join(artifacts_dir, "example_scatter_plot.png")
    plt.savefig(plot_path)
    return {"example_scatter_plot_artifact": plot_path}

model_artifact_uri = mlflow.get_artifact_uri("custom_mlflow_pyfunc")

# After training and logging both the baseline and the (custom) model
# We define a thresholds dictionary which 
thresholds = {
    "mean_squared_error": MetricThreshold(
        threshold=0.6,  # Maximum MSE threshold to accept the model, so we require MSE < 0.6
        min_absolute_change=0.1,  # Minimum absolute improvement compared to baseline
        min_relative_change=0.05,  # Minimum relative improvement compared to baseline
        greater_is_better=False  # Lower MSE is better
    )
}

baseline_model_artifact_uri = mlflow.get_artifact_uri("baseline_mlflow_pyfunc")

mlflow.evaluate(
    model_artifact_uri,
    test,
    targets="quality",
    model_type="regressor",
    evaluators=["default"],
    custom_metrics=[
        squared_diff_plus_one_metric,
        sum_on_target_divided_by_two_metric
    ],
    custom_artifacts=[prediction_target_scatter],
    validation_thresholds=thresholds,
    baseline_model=baseline_model_artifact_uri
)
```

## Model Registry Component

A model registry is a central database where model versions are stored along with their metadata; additionally, we have a UI and APIs to the registry. The model artifacts stay where they are after logged; only the reference to it is stored, along with the metadata. The registered models can be see in the **Models** menu (horizontal menu).

Pre-requisites:

- Start a server `mlflow server ...` and then `mlflow.set_tracking_uri()` in the code. In my tests, if I start the UI with `mlflow ui` it also works by using `uri="http://127.0.0.1:5000/"`; however, note that depending on where/how we start the server, the `mlruns` folder is placed in different locations...
- Log the model.

We have several options to register a model:

- After we have logged the model, in the UI: Select experiment, Artifacts, Select model, Click on **Register** (right): New model, write name; the first time we need to write a model name. The next times, if the same model, we choose its name, else we insert a new name. If we register a new model with the same name, its version will be changed.
- In the `log_model()`, if we pass the parameter `registered_model_name`.
- By calling `register_model()`.

In the **Models** menu (horizontal menu), we all the regstered model versions:

- We can/should add descriptions.
- We should add tags: which are production ready? was a specific data slice used?
- We can can tags and descriptions at model and version levels: that's important!

We can use:

- Tags: we can manually tag model versions to be `staging, production, archive`.
- Aliases: named references for particular model versions; for example, setting a **champion** alias on a model version enables you to fetch the model version by that alias via the `get_model_version_by_alias()` client API or the model URI `models:/<registered model name>@champion`.

### API Calls

Via `mlflow.sklearn.log_model`:

```python
# ...

# Connect to the tracking server
# Make sure a server is started with the given URI: mlflow server ...
mlflow.set_tracking_uri(uri="http://127.0.0.1:5000/")

# ...

# The first time, a version 1 is going to be created, then 2, etc.
# We could in theory log a model which has been trained outside from a run
mlflow.sklearn.log_model(
  model,
  "model",
  registered_model_name='elastcinet-api'
)
```

Via `mlflow.sklearn.register_model`:

```python
# ...

# Connect to the tracking server
# Make sure a server is started with the given URI: mlflow server ...
mlflow.set_tracking_uri(uri="http://127.0.0.1:5000/")

# ...

# We still need to log themodel, but wince we use register_model
# here no registered_model_name is passed!
mlflow.sklearn.log_model(lr, "model")

# The model_uri can be a path or a URI, e.g., runs:/...
# but no models:/ URIs are accepted currently
# As with registered_model_name, the version is automatically increased
mlflow.register_model(
    model_uri=f'runs:/{run.info.run_id}/model',
    name='elastic-api-2',
    tags={'stage': 'staging', 'preprocessing': 'v3'}
)

# We can load the registered model
# by using an URI like models:/<model_name>/<version>
ld = mlflow.pyfunc.load_model(model_uri="models:/elastic-api-2/1")
predicted_qualities=ld.predict(test_x)
```

## Project Component

A more extensive overview on the Project Component is given in these other notes of mine: [MLops Udacity - Reproducible Pipelines](https://github.com/mxagar/mlops_udacity/blob/main/02_Reproducible_Pipelines/MLOpsND_ReproduciblePipelines.md). While the current guide focuses on tracking and model handling, the Udacity notes focus more on how the project pipelines can be built using MLflow. Among others, sophisticated pipelines can be defined so that several components/modules are run one after the other, each storing artifacts used by the one that come later.

MLflow Projects works with a `MLproject` YAML file placed in the project folder; this configuration file contains information about

- the name of the package/project module,
- the environment with the dependencies,
- and the entry points, with their parameters.

```yaml
name: "Elastice Regression project"
conda_env: conda.yaml

entry_points:
  main:
    command: "python main.py --alpha={alpha} --l1_ratio={l1_ratio}"
    parameters:
      alpha:
        type: float
        default: 0.4

      l1_ratio:
        type: float
        default: 0.4

```

An `MLproject` can contain several **entry points**, which are basically points from which the execution of the package starts. Entry points have:

- A name; if one, entry point, usually it's called main.
- One command, which contains placeholders replaced by the parameter values. The command is the execution call of a script, which should usually parse arguments, i.e., parameter values.
- The parameters replaced in the commad; they consist of a name, a default value and a type (4 possible: string, float, path, URI); types are validated.
- An optional environment (e.g., conda), specific for the entry point command execution.

We can run the project entry points in two ways:

- Via CLI: `mlflow run [OPTIONS] URI`
  - `URI` can be a local path (e.g., `.`) or a URI to a remote machine, a git repository, etc.
  - The `OPTIONS` depend on how we run the project (locally, remotely, etc.); see list below.
- Or within a python module/code: `mlflow.projects.run()`

### Running Projects via CLI

```bash
cd ... # we go to the folder where MLproject is
# Since the only entry point is main, we don't need to specify it (because main is the default)
# We could try further options, e.g., --experiment-name, etc. See README for more.
mlflow run -P alpha=0.3 -P l1_ratio=0.3 .
# The environment will be installed
# The script from the main entrypoint will be run
# Advantage wrt. simply running the script: we can run remote scripts
```

### Running Projects via the Python API

The file [`09_projects/run.py`](./examples/09_projects/run.py) shows how to execute the `mlflow run` froma Python script:

```python
import mlflow

parameters={
    "alpha":0.3,
    "l1_ratio":0.3
}

experiment_name = "Project exp 1"
entry_point = "main"

mlflow.projects.run(
    uri=".",
    entry_point=entry_point,
    parameters=parameters,
    experiment_name=experiment_name
)
```

To use it:

```bash
conda activate mlflow
cd ...
python run.py
```

## MLflow Client

MLflow provides a class [`MlflowClient`](https://www.mlflow.org/docs/latest/python_api/mlflow.client.html?highlight=client#module-mlflow.client), which facilitates makes possible to programmatically interact with its objects in a unified form. 

- Experiment management
- Run management and tracking
- Model versioning and management

However, `MlflowClient` does not replace the MLflow library, but it provides extra functionalities to handle the tracking server object. **It provides some of the functionalities of the UI, but via code**.

The file [`10_client/client_management.py`](./examples/10_client/client_management.py) shows the most important calls to manage MLflow objects via the Python API using `mlflow.client.MlflowClient`:

- Experiments: creating, adding tags, renaming, getting and searching experiments, deleting, restoring.
- Runs: creating, renaming, settng status getting and searching runs, deleting, restoring.
- Logging/extracting parameters, metrics and artifacts via the client.
- Creating and registering model versions, setting tags, searching and getting models, deleting.

## CLI Commands

MLflow has an extensive set of [CLI tools](https://mlflow.org/docs/latest/cli.html).

First, we need to start a server.

```bash
conda activate mlflow
cd ...
mlflow server
```

Then, in another terminal, we can use the [MLflow CLI](https://mlflow.org/docs/latest/cli.html):

```bash
conda activate mlflow
cd ...
# We might need to set the environment variable
# MLFLOW_TRACKING_URI="http://127.0.0.1:5000"

# Check configuration to see everything is correctly set up
# We get: Python & MLflow version, URIs, etc.
mlflow doctor

# Use --mask-envs to avoid showing env variable values, in case we have secrets
mlflow doctor --mask-envs

# List artifacts
# We can also use the artifact path instead of the run_id
mlflow artifacts list --run-id <run_id>
mlflow artifacts list --run-id 58be5cac691f44f98638f03550ac2743

# Download artifacts
# --dst-path: local path to which the artifacts are downloaded (created if inexistent)
mlflow artifacts download --run-id <run_id> --dst-path cli_artifact
mlflow artifacts download --run-id 58be5cac691f44f98638f03550ac2743 --dst-path cli_artifact

# Upload/log artifacts
# --local-dir: local path where the artifact is
# --artifact-path: the path of the artifact in the mlflow data system
mlflow artifacts log-artifacts --local-dir cli_artifact --run-id <run_id> --artifact-path cli_artifact

# Upgrade the schema of an MLflow tracking database to the latest supported version
mlflow db upgrade sqlite:///mlflow.db

# Download to a local CSV all the runs (+ info) of an experiment
mlflow experiments csv --experiment-id <experiment_id> --filename experiments.csv
mlflow experiments csv --experiment-id 932303397918318318 --filename experiments.csv

# Create experiment; id is returned
mlflow experiments create --experiment-name cli_experiment # experiment_id: 794776876267367931

mlflow experiments rename --experiment-id <experiment_id> --new-name test1

mlflow experiments delete --experiment-id <experiment_id>

mlflow experiments restore --experiment-id <experiment_id>

mlflow experiments search --view "all" 

mlflow experiments csv --experiment-id <experiment_id> --filename test.csv

# List the runs of an experiment
mlflow runs list --experiment-id <experiment_id> --view "all"
mlflow runs list --experiment-id 932303397918318318 --view "all"

# Detailed information of a run: JSON with all the information returned
mlflow runs describe --run-id <run_id>
mlflow runs describe --run-id 58be5cac691f44f98638f03550ac2743

mlflow runs delete --run-id 

mlflow runs restore --run-id 
```

## Integration with AWS SageMaker

See dedicated section in [`README.md`](./README.md).
