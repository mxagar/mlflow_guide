# MLflow

These are my personal notes on how to use MLflow, compiled after following courses and tutorials, as well as making personal experiences.

**The main course I followed to structure the guide is [MLflow in Action - Master the art of MLOps using MLflow tool](https://www.udemy.com/course/mlflow-course), created by J Garg and published on Udemy.**

I also followed the official MLflow tutorials as well as other resources; in any case, these are all referenced.

In addition to the current repository, you might be interested in my notes on the Udacity ML DevOps Nanodegree, which briefly introduces MLflow, [mlops_udacity](https://github.com/mxagar/mlops_udacity):

- [Reproducible Model Workflows](https://github.com/mxagar/mlops_udacity/blob/main/02_Reproducible_Pipelines/MLOpsND_ReproduciblePipelines.md)
- [Deploying a Scalable ML Pipeline in Production](https://github.com/mxagar/mlops_udacity/blob/main/03_Deployment/MLOpsND_Deployment.md)

## Overview

- [MLflow](#mlflow)
  - [Overview](#overview)
  - [1. Introduction to MLOps](#1-introduction-to-mlops)
  - [2. Introduction to MLflow](#2-introduction-to-mlflow)
    - [Components](#components)
    - [Setup](#setup)
  - [3. MLflow Tracking Component](#3-mlflow-tracking-component)
    - [Basic Tracking - 01\_tracking](#basic-tracking---01_tracking)
    - [MLflow UI - 01\_tracking](#mlflow-ui---01_tracking)
    - [Extra: MLflow Tracking Quickstart with Server, Model Registration and Loading](#extra-mlflow-tracking-quickstart-with-server-model-registration-and-loading)
  - [4. MLflow Logging Functions](#4-mlflow-logging-functions)
    - [Get and Set Tracking URI - 02\_logging](#get-and-set-tracking-uri---02_logging)
    - [Experiment: Creating and Setting - 02\_logging](#experiment-creating-and-setting---02_logging)
    - [Runs: Starting and Ending - 02\_logging](#runs-starting-and-ending---02_logging)
    - [Logging Parameters, Metrics, Artifacts and Tags](#logging-parameters-metrics-artifacts-and-tags)
  - [5. Launch Multiple Experiments and Runs - 03\_multiple\_runs](#5-launch-multiple-experiments-and-runs---03_multiple_runs)
  - [6. Autologging in MLflow - 04\_autolog](#6-autologging-in-mlflow---04_autolog)
  - [7. Tracking Server of MLflow](#7-tracking-server-of-mlflow)
  - [8. MLflow Model Component](#8-mlflow-model-component)
    - [Storage Format: How the Models are Packages and Saved](#storage-format-how-the-models-are-packages-and-saved)
    - [Model Signatures - 05\_signatures](#model-signatures---05_signatures)
    - [Model API](#model-api)
  - [9. Handling Customized Models in MLflow](#9-handling-customized-models-in-mlflow)
    - [Example: Custom Python Model - 06\_custom\_libraries](#example-custom-python-model---06_custom_libraries)
    - [Custom Flavors](#custom-flavors)
  - [10. MLflow Model Evaluation](#10-mlflow-model-evaluation)
  - [11. MLflow Registry Component](#11-mlflow-registry-component)
  - [12. MLflow Project Component](#12-mlflow-project-component)
  - [13. MLflow Client](#13-mlflow-client)
  - [14. MLflow CLI Commands](#14-mlflow-cli-commands)
  - [15. AWS Integration with MLflow](#15-aws-integration-with-mlflow)
  - [Authorship](#authorship)
  - [Interesting Links](#interesting-links)

The examples are located in [`examples/`](./examples/).

## 1. Introduction to MLOps

See [Building a Reproducible Model Workflow](https://github.com/mxagar/mlops_udacity/blob/main/02_Reproducible_Pipelines/MLOpsND_ReproduciblePipelines.md).

## 2. Introduction to MLflow

[MLflow](https://mlflow.org/docs/latest/index.html) was created by Databricks in 2018 and keeps being maintained by them; as they describe it...

> MLflow is an open-source platform, purpose-built to assist machine learning practitioners and teams in handling the complexities of the machine learning process. 
> MLflow focuses on the full lifecycle for machine learning projects, ensuring that each phase is manageable, traceable, and reproducible.
> MLflow provides a unified platform to navigate the intricate maze of model development, deployment, and management.

Main MLflow components:

- Tracking: track experiements and compare parameters and results/metrics.
- Projects: package code to ensure reusability and reproducibility.
- Model and model registry: packaging for deployment, storing, and reusing models.

Additional (newer) components:

- MLflow Deployments for LLMs
- Evaluate
- Prompt Engineering UI
- Recipes

MLflow...

- is Language agnostic: it is a modular API-first approach, can be used with any language and minor changes are required in our code.
- is Compatible: can be used in combination with any ML library/framework (PyTorch, Keras/TF, ...).
- supports Integration tools: Docker containers, Spark, Kubernetes, etc.

### Components

As mentioned, the main/foundational components are:

- Tracking
- Projects
- Model
- Model registry

Other points:

- Local and remote tracking servers can be set.
- There is a UI.
- There is a CLI.
- Packaged models support many framework-model *flavours*, and can be served in varios forms, such as docker containers and REST APIs.

### Setup

In order to use MLflow, we need to set up a Python environment and install MLflow using the [`requirements.txt`](./requirements.txt) file; here a quick recipe with the [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) environment manager and [pip-tools](https://github.com/jazzband/pip-tools):

```bash
# Set proxy, if required

# Create environment, e.g., with conda, to control Python version
conda create -n mlflow python=3.10 pip
conda activate mlflow

# Install pip-tools
python -m pip install -U pip-tools

# Generate pinned requirements.txt
pip-compile requirements.in

# Install pinned requirements, as always
python -m pip install -r requirements.txt

# If required, add new dependencies to requirements.in and sync
# i.e., update environment
pip-compile requirements.in
pip-sync requirements.txt
python -m pip install -r requirements.txt

# To delete the conda environment, if required
conda remove --name mlflow --all
```

## 3. MLflow Tracking Component

### Basic Tracking - 01_tracking

MLflow distinguishes:

- Experiments: logical groups of runs
- Runs: an experiment can have several runs, which is a single code execution,
  - each with a defined set of hyperparameters, which can be specific to the run and a specific code version,
  - and where run metrics can be saved.

In the section example, a regularized linear regression is run using `ElasticNet` from `sklearn` (it combines L1 and L2 regularizations).

Summary of [`01_tracking/basic_regression_mlflow.py`](./examples/01_tracking/basic_regression_mlflow.py):

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
    mlflow.sklearn.log_model(
        sk_model=lr,
        artifact_path="wine_model", # dir name in the artifacts to dump model
        signature=signature,
        input_example=train_x[:2],
        # If registered_model_name is given, the model is registered!
        #registered_model_name=f"elasticnet-{alpha}-{l1_ratio}",
    )
```

We can run the script as follows:

```bash
conda activate mlflow
cd .../examples/01_tracking
# Run 1
python ./basic_regression_mlflow.py # default parameters: 0.7, 0.7
# Run 2
python ./basic_regression_mlflow.py --alpha 0.5 --l1_ratio 0.1
# Run 3
python ./basic_regression_mlflow.py --alpha 0.1 --l1_ratio 0.9
```

Then, a folder `mlruns` is created, which contains all the information of the experiments we create and the associated runs we execute.

This `mlruns` folder is very important, and it contains the following

```
.trash/             # deleted infor of experiments, runs, etc.
0/                  # default experiment, ignore it
99xxx/              # our experiment, hashed id
  meta.yaml         # experiment YAML: id, name, creation time, etc.
  8c3xxx/           # a run, for each run we get a folder with an id
    meta.yaml       # run YAML: id, name, experiment_id, time, etc.
    artifacts/
      mymodel/      # dumped model: PKL, MLmodel, conda.yaml, requirements.txt, etc.
        ...
    metrics/        # once ASCII file for each logged metric
    params/         # once ASCII file for each logged param
    tags/           # metadata tags, e.g.: run name, committ hash, filename, ...
  6bdxxx/           # another run
    ...
models/             # model registry, if we have registered any model
  <model_name>/
    meta.yaml
    version-x/
      meta.yaml
```

 Notes:

- We can specify where this `mlruns` folder is created.
- Usually, the `mlruns` folder should be in a remote server; if local, we should add it to `.gitignore`.
- **Note that `artifacts/` contains everything necessary to re-create the environment and load the trained model!**
- **We have logged the model, but it's not registered unless the parameter `registered_model_name` is passed, i.e., there's no central model registry yet without the registering name!**
- Usually, the UI is used to visualize the metrics; see below.

### MLflow UI - 01_tracking

The results of executing different runs can be viewed in a web UI:

```bash
conda activate mlflow
# Go to the folder where the experiment/runs are, e.g., we should see the mlruns/ folder
cd .../examples/01_tracking
# Serve web UI
mlflow ui
# Open http://127.0.0.1:5000 in browser
# The experiments and runs saved in the local mlruns folder are loaded
```

The UI has two main tabs: `Experiments` and ``Models`.

In `Models`, we can see the registered models.

In `Experiments`, we can select our `experiment_1` and run information is shown:

- We see each run has a (default) name assigned, if not given explicitly.
- Creation time stamp appears.
- We can add param/metric columns.
- We can filter/sort with column values.
- We can select Table/Chart/Evaluation views.
- We can download the runs as a CSV.
- We can select >= 2 runs and click on `Compare`; different comparison plots are possible: 
  - Parallel plot
  - Scatter plot
  - Box plot
  - Contour plot
- We can click on each run and view its details:
  - Parameters
  - Metrics
  - Artifacts: here we see the model and we can register it if we consider the run produced a good one.

![MLflow Experiments: UI](./assets/mlflow_experiments_ui.jpg)

![MLflow Experiments Plots: UI](./assets/mlflow_experiments_ui_plots.jpg)

![MLflow Experiment Run: UI](./assets/mlflow_experiments_ui_run.jpg)

### Extra: MLflow Tracking Quickstart with Server, Model Registration and Loading

Source: [MLflow Tracking Quickstart](https://mlflow.org/docs/latest/getting-started/intro-quickstart/index.html)

In addition to the example above, this other (official) example is also interesting: The Iris dataset is used to fit a logistic regression. These new points are shown:

- A dedicated server is started with `mlflow server`; beforehand, we did not explicitly start a server, i.e., the library operated without any servers. We can start a server to, e.g., have a local/remote server instance. In the following example, a local server is started. In those cases, we need to explicitly use the server URI in the code. Additionally, since we now have a server, we don't run `mlflow ui`, but we simply open the server URI.
- MLflow tracking/logging is done using the server URI.
- The model is loaded using `mlflow.pyfunc.load_model()` and used to generate some predictions.

A server is created as follows:

```bash
mlflow server --host 127.0.0.1 --port 8080
# URI: http://127.0.0.1:8080, http://localhost:8080
# To open the UI go to that URI with the browser
```

Even though for the user starting or not starting the server seems to have minimal effects on the operations (only the URI needs to be set), the underlying architecture is different:

- When no server is launched, `mlflow` is used as a library which creates/stores some files.
- When a server is launched, the `mlflow` library communicates to a server (REST) which creates/stores some files.

For more information on the **tracking server**, see the section [7. Tracking Server of MLflow](#7-tracking-server-of-mlflow).

Example code:

1. ML training and evaluation
2. MLflow tracking with model registration
3. MLflow model loading and using

```python
import mlflow
from mlflow.models import infer_signature

import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

### -- 1. ML Training and evaluation

# Load the Iris dataset
X, y = datasets.load_iris(return_X_y=True)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define the model hyperparameters
params = {
    "solver": "lbfgs",
    "max_iter": 1000,
    "multi_class": "auto",
    "random_state": 8888,
}

# Train the model
lr = LogisticRegression(**params)
lr.fit(X_train, y_train)

# Predict on the test set
y_pred = lr.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)

### -- 2. MLflow tracking with model registration

# Set our tracking server uri for logging
mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

# Create a new MLflow Experiment
mlflow.set_experiment("MLflow Quickstart")

# Start an MLflow run
with mlflow.start_run():
    # Log the hyperparameters
    mlflow.log_params(params)

    # Log the loss metric
    mlflow.log_metric("accuracy", accuracy)

    # Set a tag that we can use to remind ourselves what this run was for
    mlflow.set_tag("Training Info", "Basic LR model for iris data")

    # Infer the model signature: model input and output schemas
    signature = infer_signature(X_train, lr.predict(X_train))

    # Log the model
    model_info = mlflow.sklearn.log_model(
        sk_model=lr,
        artifact_path="iris_model",
        signature=signature,
        input_example=X_train,
        registered_model_name="tracking-quickstart",
    )

### -- 3. MLflow model loading and using

# Load the model back for predictions as a generic Python Function model
loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)

predictions = loaded_model.predict(X_test)

iris_feature_names = datasets.load_iris().feature_names

result = pd.DataFrame(X_test, columns=iris_feature_names)
result["actual_class"] = y_test
result["predicted_class"] = predictions

print(result[:4])

```

## 4. MLflow Logging Functions

In this section `mlflow.log_*` functions are explained in detail.

### Get and Set Tracking URI - 02_logging

We can use MLflow tracking in different ways:

- If we simply write python code, `mlruns` is created locally and all information is stored there. Then, we start `mlflow ui` in the terminal, in the folder which contains `mlruns`, to visualize the results.
- We can also run `mlflow server --host <HOST> --port <PORT>`; in that case, in our code we need to `mlflow.set_tracking_uri(http://<HOST>:<PORT>)` to connect to the tracking server and to open the UI we need to open `http://<HOST>:<PORT>` with the browser.
- Additionally, we can use `set_tracking_uri()` to define in the code where the data is/should be stored. Similarly, `get_tracking_uri()` retrieves the location.

Possible parameter values for `set_tracking_uri()`

    empty string: data saved automatically in ./mlruns
    local folder name: "./my_folder"
    file path: "file:/Users/username/path/to/file" (no C:)
    URL:
      (local) "http://localhost:5000"
      (remote) "https://my-tracking-server:5000"
    databricks workspace: "databricks://<profileName>"

The file [`02_logging/uri.py`](./examples/02_logging/uri.py) is the same as [`01_tracking/basic_regression_mlflow.py`](./examples/01_tracking/basic_regression_mlflow.py), but with these new lines:

```python
# We set the empty URI
mlflow.set_tracking_uri(uri="")
# We get the URI
print("The set tracking uri is ", mlflow.get_tracking_uri()) # ""
# Create experiment, if not existent, else set it
exp = mlflow.set_experiment(experiment_name="experment_1")
```

If we change:

- `uri="my_tracking"`
- `experiment_name="experiment_2"`

Then, we're going to get a new folder `my_tracking` beside usual `mlruns`.

We can run the script as follows:

```bash
conda activate mlflow
cd .../examples/02_logging
python ./uri.py

# To start the UI pointing to that tracking folder
mlflow ui --backend-store-uri 'my_tracking'
```

### Experiment: Creating and Setting - 02_logging

Original MLflow documentation:

- [Creating Experiments](https://mlflow.org/docs/latest/getting-started/logging-first-model/step3-create-experiment.html)
- [`mlflow.create_experiment()`](https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.create_experiment)
- [`mlflow.set_experiment()`](https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.set_experiment)

The file [`02_logging/experiment.py`](./examples/02_logging/experiment.py) is the same as [`01_tracking/basic_regression_mlflow.py`](./examples/01_tracking/basic_regression_mlflow.py), but with these new lines:

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

### Runs: Starting and Ending - 02_logging

We can start runs outside from `with` contexts.

Original documentation links:

- [`mlflow.start_run()`](https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.start_run)
- [`mlflow.end_run()`](https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.end_run)
- [`mlflow.active_run()`](https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.active_run)
- [`mlflow.last_active_run()`](https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.last_active_run)

The file [`02_logging/run.py`](./examples/02_logging/run.py) is the same as [`01_tracking/basic_regression_mlflow.py`](./examples/01_tracking/basic_regression_mlflow.py), but with these new lines:

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

### Logging Parameters, Metrics, Artifacts and Tags

We have several options to log parameters, metrics and artifacts, as shown below.

The file [`02_logging/artifact.py`](./examples/02_logging/artifact.py) is similar to [`01_tracking/basic_regression_mlflow.py`](./examples/01_tracking/basic_regression_mlflow.py); these are the 

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

## 5. Launch Multiple Experiments and Runs - 03_multiple_runs

In some cases we want to do several runs in the same training session:

- When we perform *incremental training*, i.e., we train until a given point and then we decide to continue doing it.
- If we are saving *model checkpoints*.
- *Hyperparameter tuning*: one run for eahc parameter set.
- *Cross-validation*: one run for each fold.
- *Feature engineering*: one run for each set of transformations.
- ...

Similarly, we can launch several experiments in a process; this makes sense when we are trying different models.

In order to run several experiments/runs one after the other, we can just choose the names of each manually, nothing more needs to be done.

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

Examples in [`03_multiple_runs/multiple_runs.py`](./examples/03_multiple_runs/multiple_runs.py).

Note that if we launch several runs and experiments, it makes sense to launch them in parallel!

## 6. Autologging in MLflow - 04_autolog

MLflow allows automatically logging parameters and metrics, without the need to specifying them explicitly. We just need to place `mlflow.autolog()` just before the model definition and training; then, all the model parameters and metrics are logged.

If we activate the autologging but would like to still log manually given things (e.g., models), we need to de-activate the autologging for those things in the `mlflow.autolog()` call.

```python
# Generic autolog: the model library is detected and its logs are carried out
mlflow.autolog(log_models: boot = True, # log model or not
               log_input_examples: bool = False, # log input examples or not
               log_model_signatures: bool = True, # signatures: schema of inputs and outputs
               log_datasets: bool = False,
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

## 7. Tracking Server of MLflow

Instead of storing everything locally on `./mlruns`, we can launch a **tracking server** hosted loally or remotely, as explained in the section [Extra; MLflow Tracking Quickstart with Server Model Registration and Loading](#extra-mlflow-tracking-quickstart-with-server-model-registration-and-loading). Then, the experiments are run on the *client*, which sends the information to the *server*.

The tracking server has 2 components:

- Storage: We have two types, and both can be local/remote:
  - **Backend store**: metadata, parameters, metrics, etc. We have two types:
    - DB Stores: SQLite, MySQL, PostgreSQL, MsSql
    - File Stores: local, Amazon S3, etc.
  - **Artifact store**: artifacts, models, images, etc. Can be also local or remote!
- Networking (communication): we stablish communucation between the client and the server. We have three types of communication:
  - **REST API (HTTP)**
  - RPC (gRPC)
  - Proxy access: restricted access depending on user/role

For small projects, we can have everythung locally; however, as the projects get larger, we should have remote/distributed architectures.

We can consider several scenarios:

1. MLflow locally:
  - client: local machine where experiments run
  - localhost:5000, but no separate server, i.e., no `mlflow server` launched
  - artifact store in `./mlruns` or a specified folder
  - backend store in `./mlruns` or a specified folder
2. MLflow locally with SQLite:
  - client: local machine where experiments run
  - localhost:5000, but no separate server, i.e., no `mlflow server` launched
  - artifact store in `./mlruns` or a specified folder
  - **backend store in SQLite or similar DB, hosted locally**
3. MLflow locally with Tracking Server
  - client: local machine where experiments run; **client connects via REST to server**
  - **localhost:5000, with separate server, i.e., launched via `mlflow server`**
  - artifact store in `./mlruns` or a specified folder
  - backend store in `./mlruns` or a specified folder
4. Remote and Distributed: MLflow with remote Tracking Server and cloud/remote storages
  - client: local machine where experiments run; **client connects via REST to server**
  - **remotehost:port, remote server launched via `mlflow server` with ports exposed**
  - **artifact store in an Amazon S3 bucket**
  - **backend store in PostgreSQL DB hosted on an another machine/node**

See all the parameters of the CLI command [`mlflow server`](https://mlflow.org/docs/latest/cli.html#mlflow-server). Here some example calls:

```bash
# Scenario 3: MLflow locally with Tracking Server
# --backend-store-uri: We specify our backend store, here a SQLite DB
# --default-artifact-root: Directory where artifacts are stored, by default mlruns, here ./mlflow-artifacts 
# --host, --port: Where the server is running, and the port; here localhost:5000
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlflow-artifacts --host 127.0.0.1 --port 5000
# Then, we can browse http.://127.0.0.1:5000
# In the exeperimentes, the tracking URI is http.://127.0.0.1:5000

# Scenario 4: Remote and Distributed: MLflow with remote Tracking Server and cloud/remote storages
mlflow server --backend-store-uri postgresql://user:password@postgres:5432/mlflowdb --default-artifact-root s3://bucket_name --host remote_host --no-serve-artifacts
```

## 8. MLflow Model Component

The MLflow Model Component allows to package models for deployment (similar to ONNX):

- Standard formats are used, along with dependencies.
- Reproducibility and reusability is enabled, by tracking lineage.
- Flexibility is allowed, by enabling realtime/online and batch inference.

Additionally, we have 

- a central repository
- and an API.

The Model Component consists of

- a **storage format**:
  - how they are packages and saved
  - all the contents in the package: metadata, version, hyperparameters, etc.
  - format itself: a directory, a single file, a Docker image, etc.
- a **signature**:
  - input and output types and shapes
  - used by the API
- the **API**:
  - REST standardized interface
  - synch / asych
  - online and batch inference
  - usable in various environments
- a [**flavor**](https://mlflow.org/docs/latest/models.html#built-in-model-flavors):
  - the serialization and storing method
  - each framework has its own methods

### Storage Format: How the Models are Packages and Saved

If we save a model locally using `mlflow.log_model()`, we'll get a local folder in the run `artifacts` with the following files:

```bash
conda.yaml            # conda environment
input_example.json    # few rows of the dataset that serve as input example
MLmodel               # YAML, most important file: model packagaing described and referenced here
model.pkl             # serialized model binary
python_env.yaml       # virtualenv 
requirements.txt      # dependencies for virtualenv
```

Those files ensure that the model environment and its environment are saved in a reproducible manner; we could set up a new environment with the same characteristics and start using the PKL.

The file `input_example.json` contains 2 rows of the input dataset:

```json
{
  "columns": ["Unnamed: 0", "fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"],
  "data": [[1316, 5.4, 0.74, 0.0, 1.2, 0.041, 16.0, 46.0, 0.99258, 4.01, 0.59, 12.5], [1507, 7.5, 0.38, 0.57, 2.3, 0.106, 5.0, 12.0, 0.99605, 3.36, 0.55, 11.4]]
}
```

`MLmodel` is the most important file and it describes the model for MLflow; really everything is defined or referenced here, which enables to reproduce the model inference anywhere:

```yaml
artifact_path: wine_model
flavors:
  python_function:
    env:
      conda: conda.yaml
      virtualenv: python_env.yaml
    loader_module: mlflow.sklearn
    model_path: model.pkl
    predict_fn: predict
    python_version: 3.10.13
  sklearn:
    code: null
    pickled_model: model.pkl
    serialization_format: cloudpickle
    sklearn_version: 1.4.1.post1
mlflow_version: 2.10.2
model_size_bytes: 1263
model_uuid: 14a531b7b86a422bbcedf78e4c23821e
run_id: 22e80d6e88a94973893abf8c862ae6ca
saved_input_example_info:
  artifact_path: input_example.json
  pandas_orient: split
  type: dataframe
signature:
  inputs: '[{"type": "long", "name": "Unnamed: 0", "required": true}, {"type": "double",
    "name": "fixed acidity", "required": true}, {"type": "double", "name": "volatile
    acidity", "required": true}, {"type": "double", "name": "citric acid", "required":
    true}, {"type": "double", "name": "residual sugar", "required": true}, {"type":
    "double", "name": "chlorides", "required": true}, {"type": "double", "name": "free
    sulfur dioxide", "required": true}, {"type": "double", "name": "total sulfur dioxide",
    "required": true}, {"type": "double", "name": "density", "required": true}, {"type":
    "double", "name": "pH", "required": true}, {"type": "double", "name": "sulphates",
    "required": true}, {"type": "double", "name": "alcohol", "required": true}]'
  outputs: '[{"type": "tensor", "tensor-spec": {"dtype": "float64", "shape": [-1]}}]'
  params: null
utc_time_created: '2024-02-27 17:14:24.719815'
```

### Model Signatures - 05_signatures

The model signature describes the data input and output types, i.e., the schema.

The types can be many, as described in [`mlflow.types.DataType`](https://mlflow.org/docs/latest/python_api/mlflow.types.html#mlflow.types.DataType). Among them, we have also `tensors`; these often

- appear when deep learning models are used,
- have one shape dimension set to `-1`, representing the batch size, which can have arbitrary values.

If the signature is saved, we can **enforce the signature**, which consists in validating the schema of the input data with the signature. This is somehow similar to using Pydantic. There are several levels of signature enforcement:

- Signature enforcement: type and name
- Name-ordering: only name order checked and fixed if necessary
- Input-type: types are checked and casted if necessary

As shown in the files [`05_signatures/manual_signature.py`](./examples/05_signatures/manual_signature.py) and [`05_signatures/infer_signature.py`](./examples/05_signatures/infer_signature.py), signatures can be defined manually or inferred automatically (preferred, recommended):

```python
from mlflow.models.signature import ModelSignature, infer_signature
from mlflow.types.schema import Schema, ColSpec

## -- Manually defined signatures (usually, not recommended)
input_data = [
    {"name": "fixed acidity", "type": "double"},
    {"name": "volatile acidity", "type": "double"},
    {"name": "citric acid", "type": "double"},
    {"name": "residual sugar", "type": "double"},
    {"name": "chlorides", "type": "double"},
    {"name": "free sulfur dioxide", "type": "double"},
    {"name": "total sulfur dioxide", "type": "double"},
    {"name": "density", "type": "double"},
    {"name": "pH", "type": "double"},
    {"name": "sulphates", "type": "double"},
    {"name": "alcohol", "type": "double"},
    {"name": "quality", "type": "double"}
]

output_data = [{'type': 'long'}]

input_schema = Schema([ColSpec(col["type"], col['name']) for col in input_data])
output_schema = Schema([ColSpec(col['type']) for col in output_data])
signature = ModelSignature(inputs=input_schema, outputs=output_schema)

input_example = {
    "fixed acidity": np.array([7.2, 7.5, 7.0, 6.8, 6.9]),
    "volatile acidity": np.array([0.35, 0.3, 0.28, 0.38, 0.25]),
    "citric acid": np.array([0.45, 0.5, 0.55, 0.4, 0.42]),
    "residual sugar": np.array([8.5, 9.0, 8.2, 7.8, 8.1]),
    "chlorides": np.array([0.045, 0.04, 0.035, 0.05, 0.042]),
    "free sulfur dioxide": np.array([30, 35, 40, 28, 32]),
    "total sulfur dioxide": np.array([120, 125, 130, 115, 110]),
    "density": np.array([0.997, 0.996, 0.995, 0.998, 0.994]),
    "pH": np.array([3.2, 3.1, 3.0, 3.3, 3.2]),
    "sulphates": np.array([0.65, 0.7, 0.68, 0.72, 0.62]),
    "alcohol": np.array([9.2, 9.5, 9.0, 9.8, 9.4]),
    "quality": np.array([6, 7, 6, 8, 7])
}

mlflow.sklearn.log_model(lr, "model", signature=signature, input_example=input_example)

## -- Automatically infered signatures (preferred, recommended)
signature = infer_signature(X_test, predicted_qualities)

input_example = {
    "columns": np.array(X_test.columns),
    "data": np.array(X_test.values)
}

mlflow.sklearn.log_model(lr, "model", signature=signature, input_example=input_example)
```

### Model API

These are the library calls to store standardized models or interact with them:

```python
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
mlflow.load_model(
  model_uri, # the model URI: /path/to/model, s3://buckect/path/to/model, etc.
  dst_path # path to download the model to
)
```

## 9. Handling Customized Models in MLflow

Custom models and custom flavors adress the use-cases in which:

- The ML library/framework is not supported by MLflow.
- We need more than the library to use our model, i.e., we have a custom Python model (with our own algorithms and libraries).

Note that:

- Custom models refer to own model libraries.
- Custom flavors refer to own model serialization methods.

### Example: Custom Python Model - 06_custom_libraries

This section works on the example files [`06_custom_libraries/model_customization.py`](./examples/06_custom_libraries/model_customization.py) and [`06_custom_libraries/load_custom_model.py`](./examples/06_custom_libraries/load_custom_model.py).

We assume that MLflow does not support Scikit-Learn, so we are going to create a Python model with it. Notes:

- We cannot use `mlflow.sklearn.log_param/metric()` functions, but instead, `mlflow.log_param/metric()`.
- We cannot use `mlflow.log_model()`, but instead `mlflow.pyfunc.log_model()`.

The way we create a custom python model is as follows:

- We dump/store all artifacts locally: dataset splits, models, etc.
- We save their paths in a dictionary called `artifacts`.
- We derive and create our own model class, based on `mlflow.pyfunc.PythonModel`.
- We create a dictionary which contains our conda environment.
- We log the model passing the last 3 objects to `mlflow.pyfunc.log_model`.
- Then, (usually in another file/run), we can load the saved model using `mlflow.pyfunc.load_model`.

These are the key parts in [`06_custom_libraries/model_customization.py`](./examples/06_custom_libraries/model_customization.py) and [`06_custom_libraries/load_custom_model.py`](./examples/06_custom_libraries/load_custom_model.py):

```python
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

When we run the script and visualize the run in the UI, we can see the following artifacts:

![Artifacts of the Custom Model](./assets/mlflow_custom_model.jpg)

More information: [MLflow: Creating custom Pyfunc models](https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#creating-custom-pyfunc-models).

### Custom Flavors

Custom flavors adress the situation in which we want to have custom serialization methods.

However, usually, that's an advanced topic which requires extending MLflow, and we are not going to need it very often.

Official docutmentation with example implementation: [Custom Flavors](https://mlflow.org/docs/latest/models.html#custom-flavors).

Necessary steps:

- Serialization and deserialization logic need to be defined.
- Create a flavor directory structure.
- Register clustom flavor.
- Define flavor-specific tools.

In practice, custom `save_model` and `load_model` functions are implemented (amongst more other) following some standardized specifications.

## 10. MLflow Model Evaluation

## 11. MLflow Registry Component

## 12. MLflow Project Component

## 13. MLflow Client

## 14. MLflow CLI Commands

## 15. AWS Integration with MLflow

## Authorship

Mikel Sagardia, 2024.  
You are free to use this guide as you wish, but please link back to the source and don't forget the original, referenced creators, which did the hardest work of compiling all the information.  
No guarantees.  

## Interesting Links

- My notes: [mlops_udacity](https://github.com/mxagar/mlops_udacity)
- [From Experiments 🧪 to Deployment 🚀: MLflow 101 | Part 01](https://medium.com/towards-artificial-intelligence/from-experiments-to-deployment-mlflow-101-40638d0e7f26)
- [From Experiments 🧪 to Deployment 🚀: MLflow 101 | Part 02](https://medium.com/@afaqueumer/from-experiments-to-deployment-mlflow-101-part-02-f386022afdc6)
- [Comprehensive Guide to MlFlow](https://towardsdatascience.com/comprehensive-guide-to-mlflow-b84086b002ae)
- [Streamline Your Machine Learning Workflow with MLFlow](https://www.datacamp.com/tutorial/mlflow-streamline-machine-learning-workflow)
- [MLOps-Mastering MLflow: Unlocking Efficient Model Management and Experiment Tracking](https://medium.com/gitconnected/mlops-mastering-mlflow-unlocking-efficient-model-management-and-experiment-tracking-d9d0e71cc697)
