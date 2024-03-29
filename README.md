# MLflow

These are my personal notes on how to use MLflow, compiled after following courses and tutorials, as well as making personal experiences.

**The main course I followed to structure the guide is [MLflow in Action - Master the art of MLOps using MLflow tool](https://www.udemy.com/course/mlflow-course), created by J Garg and published on Udemy.**

I also followed the official MLflow tutorials as well as other resources; in any case, these are all referenced.

In addition to the current repository, you might be interested in my notes on the Udacity ML DevOps Nanodegree, which briefly introduces MLflow, [mlops_udacity](https://github.com/mxagar/mlops_udacity):

- [Reproducible Model Workflows](https://github.com/mxagar/mlops_udacity/blob/main/02_Reproducible_Pipelines/MLOpsND_ReproduciblePipelines.md). While the current guide focuses on tracking and model handling, the Udacity notes focus more on how the project pipelines can be built using MLflow. Among others, sophisticated pipelines can be defined so that several components/modules are run one after the other, each storing artifacts used by the one that come later.
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
    - [Example: Evaluation of a Python Model - 07\_evaluation](#example-evaluation-of-a-python-model---07_evaluation)
    - [Example: Custom Evaluation Metrics and Artifacts - 07\_evaluation](#example-custom-evaluation-metrics-and-artifacts---07_evaluation)
    - [Example: Evaluation against Baseline - 07\_evaluation](#example-evaluation-against-baseline---07_evaluation)
  - [11. MLflow Registry Component](#11-mlflow-registry-component)
    - [Registering via UI](#registering-via-ui)
    - [Registering via API - 08\_registry](#registering-via-api---08_registry)
  - [12. MLflow Project Component](#12-mlflow-project-component)
    - [CLI Options and Environment Variables](#cli-options-and-environment-variables)
    - [Example: Running a Project with the CLI - 09\_projects](#example-running-a-project-with-the-cli---09_projects)
    - [Example: Running a Project with the Python API - 09\_projects](#example-running-a-project-with-the-python-api---09_projects)
    - [More Advanced Project Setups](#more-advanced-project-setups)
  - [13. MLflow Client](#13-mlflow-client)
  - [14. MLflow CLI Commands](#14-mlflow-cli-commands)
  - [15. AWS Integration with MLflow](#15-aws-integration-with-mlflow)
    - [AWS Account Setup](#aws-account-setup)
    - [Setup AWS CodeCommit, S3, and EC2](#setup-aws-codecommit-s3-and-ec2)
    - [Code Respository and Development](#code-respository-and-development)
      - [Data Preprocessing](#data-preprocessing)
      - [Training](#training)
      - [MLproject file and Running Locally](#mlproject-file-and-running-locally)
    - [Setup AWS Sagemaker](#setup-aws-sagemaker)
    - [Training on AWS Sagemaker](#training-on-aws-sagemaker)
    - [Model Comparison and Evaluation](#model-comparison-and-evaluation)
    - [Deployment on AWS Sagemaker](#deployment-on-aws-sagemaker)
    - [Model Inference](#model-inference)
    - [Clean Up](#clean-up)
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

- Even though here it's not obvious, MLflow works with a client-server architecture: the server is a tracking server, which can be remote, and we use a local client that can send/get the data to/from the server backend. The client is usually our program where we call the Python API.
- We can specify where this `mlruns` folder is created, it can be even a remote folder.
- If we log the artifacts using some calls, the folder `mlartifacts` might be created, close to `mlruns`; then, the artifacts are stored in `mlartifacts` and they end up being referenced in the `mlruns` YAML with a URI of the form `mlflow-artifacts:/xxx`.
- Usually, the `mlruns` and `mlartifacts` folders should be in a remote server; if local, we should add it to `.gitignore`.
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

# ...

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
  await_registration_for # wait seconds until saving ready version
)

# Load both the logged/saved model
# If the model is registered (see model registry section), we can use the URI models:/<name>/<version>
mlflow.load_model(
  model_uri, # the model URI: /path/to/model, s3://buckect/path/to/model, models:/<name>/<version>, etc.
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

MLflow provides evaluation functinalities for MLflow packaged models, i.e., we don't need to evaluate the models using other tools. The advantage is that we get

- performance metrics
- plots
- explanations (feature importance, SHAP)
  
with few coding lines, and all those eveluation data are logged. Note: **it works with the python_function (pyfunc) flavor**.

Official documentation:

- [Model Evaluation](https://mlflow.org/docs/latest/models.html#model-evaluation).
- [`mlflow.evaluate()`](https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.evaluate).

The `mlflow.evaluate()` function has the following parameters:

```python
mlflow.evaluate(
  model=None, # mlflow.pyfunc.PythonModel or model URI
  data=None, # evaluation data: np.ndarray, pd.DataFrame, PySpark DF, mlflow.data.dataset.Dataset
  model_type=None, # 'regressor', 'classifier', 'question-answering', 'text', 'text-summarization'
  targets=None, # list of evaluation labels
  dataset_path=None, # path where data is stored
  feature_names=None, # np.ndarray, pd.DataFrame, PySpark DF
  evaluators=None, # list of evaluator names, e.g., 'defaults'; all used by default - get all with mlflow.models.list_evaluators()
  evaluator_config=None, # config dict for evaluators: log_model_explainability, explainability_nsmaples, etc.
  custom_metrics=None, # list custom defined EvaluationMetric objects
  custom_artifacts=None, # list of custom artifact functions: dict->JSON, pd.DataFrame->CSV
  validation_thresholds=None, # dictionary with custom thresholds for metrics
  baseline_model=None, # baseline model to compare against
  env_manager='local', # env manager to load models in isolated Python envs: 'local' (current env), 'virtualenv' (recommended), 'conda'
  # More:
  predictions=None, extra_metrics=None,  model_config=None, baseline_config=None, inference_params=None
)
```

The `'default'` evaluator uses `shap` and we need to manually `pip install shap`. 

### Example: Evaluation of a Python Model - 07_evaluation

An evaluation example is given in [`07_evaluation/evaluate.py`](./examples/07_evaluation/evaluate.py); the `mlflow.evaluate()` call is summarized here:

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

After running the evaluation, in the artifacts, we get the SHAP plots as well as the explainer model used.

![SHAP Summary Plot](./assets/shap_summary_plot.png)


### Example: Custom Evaluation Metrics and Artifacts - 07_evaluation

We can create custom evaluation metrics and evaluation artifacts.
To that end:

- We create metric computation functions passed to `make_metric`, which creates metric measurement objects.
- Similarly, we define metric artifact computation functions (e.g., plots).
- We pass all of that to `mlflow.evaluate()` in its parameters.

As a result, we will get additional metrics in the DB or extra plots in the artifacts.

In the following, the most important lines from the example in [`07_evaluation/custom_metrics.py`](./examples/07_evaluation/custom_metrics.py):

```python
from mlflow.models import make_metric

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

# Custom metrics are created passing a custom defined function
# to make_metric. We pass to each custom defined function these parameters (fix name):
# - eval_df: the data
# - builtin_metric: a dictionry with the built in metrics from mlflow
# Note: If the args are not used, precede with _: _builtin_metrics, _eval_df
# else: builtin_metrics, eval_df
def squared_diff_plus_one(eval_df, _builtin_metrics):
    return np.sum(np.abs(eval_df["prediction"] - eval_df["target"] + 1) ** 2)

def sum_on_target_divided_by_two(_eval_df, builtin_metrics):
    return builtin_metrics["sum_on_target"] / 2

# In the following we create the metric objects via make_metric
# and passing the defined functions
squared_diff_plus_one_metric = make_metric(
    eval_fn=squared_diff_plus_one,
    greater_is_better=False, # low metric value is better
    name="squared diff plus one"
)

sum_on_target_divided_by_two_metric = make_metric(
    eval_fn=sum_on_target_divided_by_two,
    greater_is_better=True,
    name="sum on target divided by two"
)

# We can also create custom artifacts.
# To that point, we simply define the function
# which creates the artifact.
# Parameters:
# - eval_df, _eval_df
# - builtin_metrics, _builtin_metrics
# - artifacts_dir, _artifacts_dir
def prediction_target_scatter(eval_df, _builtin_metrics, artifacts_dir):
    plt.scatter(eval_df["prediction"], eval_df["target"])
    plt.xlabel("Targets")
    plt.ylabel("Predictions")
    plt.title("Targets vs. Predictions")
    plot_path = os.path.join(artifacts_dir, "example_scatter_plot.png")
    plt.savefig(plot_path)
    return {"example_scatter_plot_artifact": plot_path}

# Now, we run the evaluation wuth custom metrics and artifacts
artifacts_uri = mlflow.get_artifact_uri(model_artifact_path)
mlflow.evaluate(
    artifacts_uri,
    test,
    targets="quality",
    model_type="regressor",
    evaluators=["default"],
    # Custom metric objects
    custom_metrics=[
        squared_diff_plus_one_metric,
        sum_on_target_divided_by_two_metric
    ],
    # Custom artifact computation functions
    custom_artifacts=[prediction_target_scatter]
)
```

### Example: Evaluation against Baseline - 07_evaluation

We can define a baseline model and compare against it in the `mlflow.evaluate()` call by checking some thresholds. The baseline model needs to be passed to `mlflow.evaluate()` along with its related artifacts, as well as a `thresholds` dictionary.

As a result, we will get additional models and artifacts (baseline).

In the following, the most important lines from the example in [`07_evaluation/validation_threshold.py`](./examples/07_evaluation/validation_threshold.py):

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

## 11. MLflow Registry Component

A model registry is a central database where model versions are stored along with their metadata; additionally, we have a UI and APIs to the registry.

The model artifacts stay where they are after logged; only the reference to it is stored, along with the metadata. The registered models can be see in the **Models** menu (horizontal menu).

Pre-requisites:

- Start a server `mlflow server ...` and then `mlflow.set_tracking_uri()` in the code. In my tests, if I start the UI with `mlflow ui` it also works by using `uri="http://127.0.0.1:5000/"`; however, note that depending on where/how we start the server, the `mlruns` folder is placed in different locations...
- Log the model.

### Registering via UI

We have several options to register a model:

- After we have logged the model, in the UI: Select experiment, Artifacts, Select model, Click on **Register** (right): New model, write name; the first time we need to write a model name. The next times, if the same model, we choose its name, else we insert a new name. If we register a new model with the same name, its version will be changed.
- In the `log_model()`, if we pass the parameter `registered_model_name`.
- By calling `register_model()`.

In the **Models** menu (horizontal menu), we all the regstered model versions:

- We can/should add descriptions.
- We should add tags: which are production ready? was a specific data slice used?
- We can can tags and descriptions at model and version levels: that's important!

In older MLflow versions, a model could be in 4 stages or phases:

- None
- Staging: candidate for production, we might want to compare it with other candidates.
- Production: model ready for production, we can deploy it.
- Archive: model taken out from production, not usable anymore; however, it still remains in the registry.

Now, those stage values are deprecated; instead, we can use:

- Tags: we can manually tag model versions to be `staging, production, archive`.
- Aliases: named references for particular model versions; for example, setting a **champion** alias on a model version enables you to fetch the model version by that alias via the `get_model_version_by_alias()` client API or the model URI `models:/<registered model name>@champion`.

### Registering via API - 08_registry

As mentioned, we can register a model in the code with two functions:

- In the `log_model()`, if we pass the parameter `registered_model_name`.
- By calling `register_model()`.

In the example [`08_registry/registry_log_model.py`](./examples/08_registry/registry_log_model.py) the first approach is used. Here are the most important lines:

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

The function [`resgister_model()`](https://www.mlflow.org/docs/latest/python_api/mlflow.html?highlight=register_model#mlflow.register_model) has the following parameters:

```python
mlflow.register_model(
  model_uri, # URI or path
  name,
  await_registration_for, # Number of seconds to wait to create ready version
  tags # dictionary of key-value pairs
)
```

In the example [`08_registry/registry_register_model.py`](./examples/08_registry/registry_register_model.py) we can see how `register_model()` is used; here are the most important lines:

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

## 12. MLflow Project Component

MLflow Projects is a component which allows to organize and share our code easily.

Interesting links:

- I have another guide in [MLops Udacity - Reproducible Pipelines](https://github.com/mxagar/mlops_udacity/blob/main/02_Reproducible_Pipelines/MLOpsND_ReproduciblePipelines.md).
- [The official guide](https://www.mlflow.org/docs/latest/projects.html).

MLflow Projects works with a `MLproject` YAML file placed in the project folder; this configuration file contains information about

- the name of the package/project module,
- the environment with the dependencies,
- and the entry points, with their parameters.

Here is an example:

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

We can **run the project/package in the CLI** as follows:

```bash
cd ... # we go to the folder where MLproject is
mlflow run -P alpha=0.3 -P l1_ratio=0.3 .
```

The [**environment can be specified**](https://www.mlflow.org/docs/latest/projects.html?highlight=mlproject#mlproject-specify-environment) in several ways:

```yaml
# Virtualenv (preferred by MLflow)
python_env: files/config/python_env.yaml

# Conda: conda env export --name <env_name> > conda.yaml
# HOWEVER: note that the conda file should be generic for all platforms
# so sometimes we need to remove some build numbers...
conda_env: files/config/conda.yaml

# Docker image: we can use a prebuilt image
# or build one with --build-image
# Environment variables like MLFLOW_TRACKING_URI are propagated (-e)
# and the host tracking folder is mounted as a volume (-v)
# We can also set volumes and env variables (copied or defined)
docker_env:
  image: mlflow-docker-example-environment:1.0 # pre-built
  # image: python:3.10 # to build with `mlflow run ... --build-image`
  # image: 012345678910.dkr.ecr.us-west-2.amazonaws.com/mlflow-docker-example-environment:7.0 # fetch from registry
  volumes: ["/local/path:/container/mount/path"]
  environment: [["NEW_ENV_VAR", "new_var_value"], "VAR_TO_COPY_FROM_HOST_ENVIRONMENT"]
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

### CLI Options and Environment Variables

Some `OPTIONS`:

```bash
# mlflow run -e <my_entry_point> <uri>
-e, --entry-point <NAME>

# mlflow run -v <absc123> <uri>
-v, --version <VERSION>

# mlflow run -P <param1=value1> -P <param2=value2> <uri>
-P, --param-list <NAME=VALUE>

# mlflow run -A <param1=value1> <uri>
-A, --docker-args <NAME=VALUE>

# Specific experiement name
--experiment-name <NAME>

# Specific experiment ID
--experiment-id <ID>

# Specify backend: local, databricks, kubernetes
-b, --backend <NAME>

# Backend config file
-c, --backend-config <FILE>

# Specific environment manager: local, virtualenv, conda
--env-manager <NAME>

# Only valid for local backend
--storage-dir <DIR>

# Specify run ID
--run-id <ID>

# Specify run name
--run-name <NAME>

# Build a new docker image
--build-image
```

In addition to the options, we also have important environment variables which can be set; if set, their values are used acordingly by default:

```bash
MLFLOW_TRACKING_URI
MLFLOW_EXPERIMENT_NAME
MLFLOW_EXPERIMENT_ID
MLFLOW_TMP_DIR # --storage-dir = storage folder for local backend
```

### Example: Running a Project with the CLI - 09_projects

The file [`09_projects/main.py`](./examples/09_projects/main.py) can be run without the `MLproject` tool as follows:

```bash
conda activate mlflow
cd ...
python ./main.py --alpha 0.3 --l1_ratio 0.3
```

However, we can use the co-located [`09_projects/MLproject`](./examples/09_projects/MLproject) and run it using `mlflow`:

```bash
conda activate mlflow
cd ...
# Since the only entry point is main, we don't need to specify it (because main is the default)
# We could try further options, e.g., --experiment-name
mlflow run -P alpha=0.3 -P l1_ratio=0.3 .
# The environment will be installed
# The script from the main entrypoint will be run
# Advantage wrt. simply running the script: we can run remote scripts
```

The main difference is that now we create a specific environment only for running the project/package/module. Additionally, we could run remote modules.

### Example: Running a Project with the Python API - 09_projects

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

### More Advanced Project Setups

Visit my notes: [MLops Udacity - Reproducible Pipelines](https://github.com/mxagar/mlops_udacity/blob/main/02_Reproducible_Pipelines/MLOpsND_ReproduciblePipelines.md). While the current guide focuses on tracking and model handling, the Udacity notes focus more on how the project pipelines can be built using MLflow. Among others, sophisticated pipelines can be defined so that several components/modules are run one after the other, each storing artifacts used by the one that come later.

## 13. MLflow Client

The MLflow client is basically the counterpart of the tracking server, i.e., it's the application where we use the API library which communicates with the server. Additionally, MLflow provides a class [`MlflowClient`](https://www.mlflow.org/docs/latest/python_api/mlflow.client.html?highlight=client#module-mlflow.client), which facilitates

- Experiment management
- Run management and tracking
- Model versioning and management

However, `MlflowClient` does not replace the MLflow library, but it provides extra functionalities to handle the tracking server object. **It provides some of the functionalities of the UI, but via code**.

The file [`10_client/client_management.py`](./examples/10_client/client_management.py) shows the most important calls to manage MLflow objects via the Python API using `mlflow.client.MlflowClient`:

- Experiments: creating, adding tags, renaming, getting and searching experiments, deleting, restoring.
- Runs: creating, renaming, settng status getting and searching runs, deleting, restoring.
- Logging/extracting parameters, metrics and artifacts via the client.
- Creating and registering model versions, setting tags, searching and getting models, deleting.

The content of the script:

```python
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet

import mlflow
from mlflow import MlflowClient
from mlflow.entities import ViewType

# We need to start the server in another Termnial first: mlflow server
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Out client object
client = MlflowClient()

# Create experiment: this can be executed only once
# The id is a string
NUM = 21
exp_name = "Experiment with client " + str(NUM)
experiment_id = client.create_experiment(
    name = exp_name,
    tags = {"version": "v1", "priority": "P1"}
)
# Delete and restore an experiment
client.delete_experiment(experiment_id=experiment_id)
client.restore_experiment(experiment_id=experiment_id)

print(f'Experiment id: {experiment_id}')

# Set a tag
client.set_experiment_tag(experiment_id=experiment_id, key="framework", value="sklearn")

# Get experiment by name or id
experiment = client.get_experiment_by_name(name="Test") # None returned if doesn't exist
if experiment is None:
    experiment = client.get_experiment(experiment_id=experiment_id)

# Rename experiment
client.rename_experiment(experiment_id=experiment_id, new_name="NEW "+exp_name)

# Get properties from experiment
print(f'Experiment name: {experiment.name}')
print(f'Experiment id: {experiment.experiment_id}')
print(f'Experiment artifact location: {experiment.artifact_location}')
print(f'Experiment tags: {experiment.tags}')
print(f'Experiment lifecycle stage: {experiment.lifecycle_stage}')

# Search experiments
# https://www.mlflow.org/docs/latest/python_api/mlflow.client.html#mlflow.client.MlflowClient.search_experiments
experiments = client.search_experiments(
    view_type = ViewType.ALL, # ACTIVE_ONLY, DELETED_ONLY, ALL
    filter_string = "tags.`version` = 'v1' AND tags.`framework` = 'sklearn'",
    order_by = ["experiment_id ASC"],
    #max_results, # maximum number of experiments we want
)

# Loop and print experiments
for exp in experiments:
    print(f"Experiment Name: {exp.name}, Experiment ID: {exp.experiment_id}")

# This function simply creates a run object
# and doesn't run any code;
# so it's different from mlflow.start_run() or mlflow.projects.run()
# Use-case: we want to prepare the run object, but not run any ML code yet
# The status of the run is UNFINISHED;
# we need to change it when finished with set_terminated or update_run
run = client.create_run(
    experiment_id = experiment_id,
    tags = {
        "Version": "v1",
        "Priority": "P1"
    },
    run_name = "run from client",
    #start_time # when we'd like to start the run, if not provided, now
)

# The created run is not active,
# so we cannot log data without explicitly activating it or passing its id to the log function
# We can log via the client object
client.log_param(
    run_id=run.info.run_id,
    key="alpha",
    value=0.3
)
client.log_metric(
    run_id=run.info.run_id,
    key="r2",
    value=0.9
)

# Similarly, we have:
# - client.log_artifact()
# - client.log_image()
# ...
client.log_artifact(
    run_id=run.info.run_id,
    local_path="../data/red-wine-quality.csv"
)

# Get a run object by its id
run = client.get_run(run_id=run.info.run_id)

# Run properties
print(f"Run tags: {run.data.tags}")
print(f"Experiment id: {run.info.experiment_id}")
print(f"Run id: {run.info.run_id}")
print(f"Run name: {run.info.run_name}")
print(f"Lifecycle stage: {run.info.lifecycle_stage}")
print(f"Status: {run.info.status}")

# Extract metrics
# NOTE: metrics always have a step and a timestamp, because they are thought for DL apps
# in which we'd like to save a metric for every batch or epoch
# Thus, even though here we have saved only one element for r2, we still get a list
# with one element, and the element is an mlflow.entities.Metric object
# https://www.mlflow.org/docs/latest/python_api/mlflow.entities.html#mlflow.entities.Metric
metric_list = client.get_metric_history(
    run_id=run.info.run_id,
    key="r2"
)

for metric in metric_list:
    print(f"Step: {metric.step}, Timestamp: {metric.timestamp}, Value: {metric.value}")

# Extract info of artifacts
artifacts = client.list_artifacts(run_id=run.info.run_id)

for artifact in artifacts:
    print(f"Artifact: {artifact.path}")
    print(f"Size: {artifact.file_size}")

# Change run status to FINISHED
# We can also use client.update_run(), but that call also can change the name of the run
client.set_terminated(
    run_id=run.info.run_id,
    status="FINISHED" # 'RUNNING', 'SCHEDULED', 'FINISHED', 'FAILED', 'KILLED'
)

run = client.get_run(run_id=run.info.run_id)
print(f"Lifecycle stage: {run.info.lifecycle_stage}")
print(f"Status: {run.info.status}")

# We can delete a run: active -> deleted
# and we can also restore it
client.delete_run(run_id=run.info.run_id)

run = client.get_run(run_id=run.info.run_id)
print(f"Lifecycle stage: {run.info.lifecycle_stage}")
print(f"Status: {run.info.status}")

client.restore_run(run_id=run.info.run_id)

run = client.get_run(run_id=run.info.run_id)
print(f"Lifecycle stage: {run.info.lifecycle_stage}")
print(f"Status: {run.info.status}")

# Freeze run_id
run_id = run.info.run_id

# We can search for runs
runs = client.search_runs(
    experiment_ids=["6", "10", "12", "14", "22"],
    run_view_type=ViewType.ACTIVE_ONLY,
    order_by=["run_id ASC"],
    filter_string="run_name = 'Mlflow Client Run'"
)

for run in runs:
    print(f"Run name: {run.info.run_name}, Run ID: {run.info.run_id}")

# Get previous run again
run = client.get_run(run_id=run_id)
print(f"Lifecycle stage: {run.info.lifecycle_stage}")
print(f"Status: {run.info.status}")

# Create a registered model
# BUT, this only creates an entry in the Models tab, there is no model yet!
client.create_registered_model(
    name="lr-model"+"-"+run_id[:3], # We should not add the run_id start, this is only for my tests...
    # These tags are set at models level, not version level
    tags={
        "framework": "sklearn",
        "model": "ElasticNet"
    },
    description="Elastic Net model trained on red wine quality dataset"
)

# To work with a model, we need to create and log one first
# Then we add it to the model registry using create_model_version()
data = pd.read_csv("../data/red-wine-quality.csv")
train, test = train_test_split(data)
train_x = train.drop(["quality"], axis=1)
train_y = train[["quality"]]
lr = ElasticNet(alpha=0.35, l1_ratio=0.3, random_state=42)
lr.fit(train_x, train_y)
artifact_path = "model"
mlflow.sklearn.log_model(sk_model=lr, artifact_path=artifact_path)

# To add a model, we do it with create_model_version()
client.create_model_version(
    name="lr-model"+"-"+run_id[:3], # We should not add the run_id start, this is only for my tests...
    source=f"runs:/{run_id}/{artifact_path}",
    # These tags are set at models level, not version level
    tags={
        "framework": "sklearn",
        "hyperparameters": "alpha and l1_ratio"
    },
    description="A second linear regression model trained with alpha and l1_ratio prameters."
)

# Set tags at version level
# An alternative: update_model_version()
client.set_model_version_tag(
    name="lr-model"+"-"+run_id[:3], # We should not add the run_id start, this is only for my tests...
    version="1",
    key="framework",
    value="sklearn"
)

# We can get model versions with
# - get_latest_version()
# - get_model_version()
# - get_model_version_by_alias()
mv = client.get_model_version(
    name="lr-model"+"-"+run_id[:3], # We should not add the run_id start, this is only for my tests...
    version="1"
)

print("Name:", mv.name)
print("Version:", mv.version)
print("Tags:", mv.tags)
print("Description:", mv.description)
print("Stage:", mv.current_stage)

# We can also SEARCH for model versions
mvs = client.search_model_versions(
    filter_string="tags.framework = 'sklearn'",
    max_results=10,
    order_by=["name ASC"]
)

for mv in mvs:
    print(f"Name {mv.name}, tags {mv.tags}")

# Delete a model version
client.delete_model_version(
    name="lr-model"+"-"+run_id[:3], # We should not add the run_id start, this is only for my tests...
    version="1"
)
```

## 14. MLflow CLI Commands

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

## 15. AWS Integration with MLflow

In this section an example project is built entirely on AWS:

- Used AWS services: CodeCommit, Sagemaker (ML), EC2 (MLflow) and S3 (storage).
- Problem/dataset: [House price prediction (regression)](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques).

The code of the project is on a different repository connected to CodeCommit. However, I have added the final version to [`examples/housing-price-aws/`](./examples/housing-price-aws/).

Architecture of the implementation:

![AWS Example Architecture](./assets/aws_example_architecture.jpg)

- We can use Github or AWS CodeCommit to host the code.
- Code development and buld tests are local.
- We push the code to the remote repository.
- MLflow server is on an EC2 instance (parameters, metrics, metadata stored in the tracking server VM).
- All the model artifacts stored in an S3 bucket.
- We will compare with the UI different model versions and select one.
- Then, deployment comes: we build a docker image and set a SageMaker endpoint.
- Once deployed, we'll test the inference.

Notes:

- Create a **non-committed** folder, e.g., `credentials`, where files with secrets and specific URIs will be saved.
- We can stop the compute services while not used to save costs (stopped compute services don't iincur costs).
  - The EC2 instance requires to manually launch the MLflow server if restarted; the local data persists.
  - The SageMaker notebook requisres to re-install any packages we have installed (e.g., mlflow); the local data persists.
- When we finish, remove all resources! See section [Clean Up](#clean-up).

### AWS Account Setup

Steps:

- First, we need to create an account (set MFA).
- Then, we log in with the root user.
- After that we create a specific IAM user. Note IAM users have 3-4 elements:
  - IAM ID
  - Username
  - Password
  - Option: MFA code, specific for the IAM user

    Username (up right) > Security credentials
    Users (left menu) > Create a new user
      Step 1: User details
        User name: mlflow-user
        Provide access to AWS Management Cosole
        I want to create an IAM user
        Choose: Custom/autogenerated password
        Uncheck: Create new PW after check-in
      Step 2: Set perimissions
        Usually we grant only the necessary permissions
        In this case, we "Attach policies directly"
          Check: AdministratorAccess
          Alternative:
            Check: SageMaker, CodeCommit, S3, etc.
        Next
      Step 3: Review and create
        Next
      Step 4: Retrieve password
        Save password
        Activate MFA, specific for the IAM user!

    We get an IAM user ID and a password
    We can now use the IAM sign-in adress to log in
      https://<IAM-ID>.signin.aws.amazon.com/console
    or select IAM an introduce the IAM ID
      IAM ID: xxx
      Username: mlflow-user
      Password: yyy
    
Now, we sign out and sign in again with the IAM user credentials.
Next, we need to create **Access Keys**:

    Sign in as IAM user
    User (up left) > Security credentials
    Select: Application running on AWS compute service
      We download the access keys for the services
      we are going to use: EC2, S3, etc.
      Confirm: I understand the recommendation
    Create access key
    Download and save securely: mlflow-user_accessKeys.csv
    IMPORTANT: redentials are shown only now!

We should create a `.env` file which contains

```
AWS_ACCESS_KEY_ID="..."
AWS_SECRET_ACCESS_KEY="..."
AWS_DEFAULT_REGION="eu-central-1"
```

Then, using `python-dotenv`, we can load these variables to the environment from any Python file, when needed:

```python
from dotenv import load_dotenv

# Load environment variables in .env:
# AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION
load_dotenv()
```

However, note that these variables are needed only locally, since all AWS environment in which we log in using the IAM role have already the credentials!

### Setup AWS CodeCommit, S3, and EC2

In this section, the basic AWS components are started manually and using the web UI. I think the overall setup is not really secure, because everything is public and can be accessed from anywhere, but I guess the focus of the course is not AWS nor security...

AWS CodeCommit Repository:

    Search: CodeCommit
    Select Region, e.g., EU-central-1 (Frankfurt)
    Create repository
      Name (unique): mlflow-housing-price-example
      Description: Example repository in which house prices are regressed while using MLflow for tracking operations.
      Create

    Get credentials
      https://docs.aws.amazon.com/codecommit/latest/userguide/setting-up-gc.html?icmpid=docs_acc_console_connect_np#setting-up-gc-iam
      Download credentials and save them securely

      Username (up right) > Security credentials
      AWS CodeCommit credentials (2nd tab, horizontal menu)
        HTTPS Git credentials for AWS CodeCommit: Generate credentials
          Download credentials: mlflow-user_codecommit_credentials.csv
          IMPORTANT: only shown once!
          Save them in secure place!

    Clone repository in local folder:
      git clone https://git-codecommit.eu-central-1.amazonaws.com/v1/repos/mlflow-housing-price-example
      The first time we need to introduce our credentials downloaded before: username, pw
      If we commit & push anything, we should see the changes in the CodeCommit repository (web UI)

AWS S3 Bucket:

    Search: S3
    Create bucket
      Region: EU-central-1 (Frankfurt)
      Bucket name (unique): houing-price-mlflow-artifacts
      Uncheck: Block all public access
        Check: Acknowledge objects public
    Leave rest default
    Create bucket

AWS EC2:

    Search: EC2
    Launch instance
      We use a minimum configuration
      Number or instances: 1
      Image: Ubuntu
      Instance Type: t2.micro
      Name: mlflow-server
      Key pair (login): needed if we want to connect to EC2 from local machine
        Create a new key pair
        Name: mlflow-server-kp
        Dafault: RSA, .pem
        Create key pair
          File downloaded: mlflow-server-kp.pem
      Network settings
        Create a security group
        Allow SSH traffic from anywhere
        Allow HTTPS traffic from interent
        Allow HTTP traffic from internet
      Rest: default
      Launch instance
    EC2 > Instances (left panel): Select instance
      When instance is running:
      Connect
      Shell opens on Web UI

In the shell, we install the dependencies and start the MLflow server.
The tool [pipenv](https://pipenv.pypa.io/en/latest/) is used to create the environment; `pipenv` is similar to `pip-tools` or `poetry`. We could connect locally using the `mlflow-server-kp.pem` we have downloaded, too.

The following commands need to be run in the EC2 instance:

```bash
# Update packages list
sudo apt update

# Install pip
sudo apt install python3-pip

# Install dependencies
sudo pip3 install pipenv virtualenv

# Create project folder
mkdir mlflow
cd mlflow

# Install dependencies
pipenv install mlflow awscli boto3 setuptools

# Start a shell with the virtualenv activated. Quit with 'exit'.
pipenv shell

# Configure AWS: Now, we need the Access Key Credentials crated in the account setup
aws configure
# AWS Access Key ID: xxx
# AWS Secret Access Key: yyy
# Default region: press Enter
# Default output: press Enter

# Start MLflow server
# Host to all: 0.0.0.0
# Set 
# - the backend store: SQLite DB
# - and the artifact store: the bucket name we have specified
mlflow server -h 0.0.0.0 --backend-store-uri sqlite:///mlflow.db --default-artifact-root s3://houing-price-mlflow-artifacts
```

After the last command, the server is launched:
    
    ...
    Listening at: http://0.0.0.0:5000
    ...

To use it, we need to expose the port `5000` of the EC2 instance:

    EC2 > Instances (left panel): Select instance
      Scroll down > Security tab (horizontal menu)
      We click on our segurity group: sg-xxx
      Edit inbound rules
        Add rule
          Type: Custom TCP
          Port: 5000
          Source: Anywhere-IPv4 (0.0.0.0)
        Save rules

Additionally, we copy to a safe place the public DNS of the EC2 instance:

    EC2 > Instances (left panel): Select instance (we go back to our instance)
      Copy the Public IPv4 DNS (instance summary dashboard), e.g.:
        ec2-<IP-number>.<region>.amazonaws.com

Now, we can open the browser and paste the DNS followed by the port number 5000; the Mlflow UI will open!

    ec2-<IP-number>.<region>.amazonaws.com:5000

We can close the window of the EC2 instance shell on the web UI; the server will continue running.

**If we stop the EC2 server to safe costs, we need to re-start it, open a terminal and restart the server again by executing the commands above. However, note that the the public DNS might change!** The artifacts will persist (they are in S3) and the backend store which is local to the EC2 instance (mlflow.db), too.

If we want to connect locally to the EC2 instance, first we make sure that the port 22 is exposed to inbound connections from anywhere (follow the same steps as for opening port 5000). Then, we can `ssh` as follows:

```bash
# Go to the folder where the PEM credentials are
cd .../examples/housing-price-aws/credentials

# On Unix, we need to make the PEM file readable only for the owner
# chmod 600 mlflow-server-kp.pem

# Connect via SSH
# Replace:
# - username: ubuntu if Ubuntu, ec2-user is Amazon Linux
# - Public-DNS: ec2-<IP-number>.eu-central-1.compute.amazonaws.com
# - key-pair.pem: mlflow-server-kp.pem
ssh -i <key-pair.pem> <username>@<Public-DNS>
```

### Code Respository and Development

From now on, we work ok the repository cloned locally.

Respository structure:

```
C:.
│   conda.yaml        # Environment
│   data.py           # Data pre-processing
│   deploy.py         # Deploy model to Sagemaker endopoint image
│   MLproject
│   params.py         # Hyperparameter search space
│   run.py            # Run training using mlflow & MLproject
│   predict.py        # Inference
│   train.py          # Entrypoint for MLproject
│   eval.py           # EValuation metrics
│
├───credentials/
│
└───data/
        test.csv
        train.csv
```

Dataset schema:

```
 #   Column         Non-Null Count  Dtype  
---  ------         --------------  -----  
 0   Id             1460 non-null   int64  
 1   MSSubClass     1460 non-null   int64  
 2   MSZoning       1460 non-null   object 
 3   LotFrontage    1201 non-null   float64
 4   LotArea        1460 non-null   int64  
 5   Street         1460 non-null   object 
 6   Alley          91 non-null     object 
 7   LotShape       1460 non-null   object 
 8   LandContour    1460 non-null   object 
 9   Utilities      1460 non-null   object 
 10  LotConfig      1460 non-null   object 
 11  LandSlope      1460 non-null   object 
 12  Neighborhood   1460 non-null   object 
 13  Condition1     1460 non-null   object 
 14  Condition2     1460 non-null   object 
 15  BldgType       1460 non-null   object 
 16  HouseStyle     1460 non-null   object 
 17  OverallQual    1460 non-null   int64  
 18  OverallCond    1460 non-null   int64  
 19  YearBuilt      1460 non-null   int64  
...
 79  SaleCondition  1460 non-null   object 
 80  SalePrice      1460 non-null   int64  <- TARGET!
```

#### Data Preprocessing

All the data preprocessing happens in `data.py`:

- Datasets loaded (train & test).
- Train/validation split.
- Target/independent variable selection.
- Missing value imputation with KNN.
- Categorical feature one-hot encoding.
- **The transformed `X_train`, `X_val` and `test` are the product.**

**IMPORTANT NOTE**: Even though the `data.py` script works with the environment, it has some issues for newer Scikit-Learn versions and the code needs to be updated accordingly: `OneHotEncoder` returns a sparse matrix and, on top pf that, we should apply it to categoricals only, thus we proably need a `ColumnTransformer`.

#### Training

The training happens in 

- `params.py`:
- `eval.py`:
- `train.py`: 

Here, we start using `mlflow`; however, the `mlflow` dependencies and calls are only in `train.py`. Additionally, those dependencies/calls are as generic as possible, i.e., we don't define any experiment/run ids/names, tracking URIs, etc. The idea is to have the code as reusable as possible, and we leave any configuration to `MLproject` and higher level files, like `run.py`:

```python
import argparse
import mlflow
from sklearn.linear_model import Ridge, ElasticNet
from xgboost import XGBRegressor
from sklearn.model_selection import ParameterGrid
from data import X_train, X_val, y_train, y_val
from params import ridge_param_grid, elasticnet_param_grid, xgb_param_grid
from eval import eval_metrics


def train(model_name):
    
    # Select model and parameter grid based on input argument
    # Dafault: XGBRegressor
    model_cls = XGBRegressor
    param_grid = xgb_param_grid
    log_model = mlflow.xgboost.log_model
    if model_name == 'ElasticNet':
        model_cls = ElasticNet
        param_grid = elasticnet_param_grid
        log_model = mlflow.sklearn.log_model
    elif model_name == 'Ridge':
        model_cls = Ridge
        param_grid = ridge_param_grid
        log_model = mlflow.sklearn.log_model
    else:
        # Defaults to XGBRegressor if --model is not provided or is incorrect
        pass

    # NOTE: We usually don't hard-code any experiment/run name/ids,
    # these are set dynamically
    # and here MLproject contains the important configuration,
    # so that the script/code is generic!
    # Also, it is common to have a run.py file which uses hard-coded values (i.e., URIs).
    # Loop through the hyperparameter combinations and log results in separate runs
    for params in ParameterGrid(param_grid):
        with mlflow.start_run():
            # Fit model
            model = model_cls(**params)
            model.fit(X_train, y_train)

            # Evaluate trained model
            y_pred = model.predict(X_val)
            metrics = eval_metrics(y_val, y_pred)

            # Logging the inputs and parameters
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)

            # Log the trained model
            log_model(
                model,
                model_name,
                input_example=X_train[:5],
                code_paths=['train.py', 'data.py', 'params.py', 'eval.py']
            )

if __name__ == "__main__":
    # Parse arguments with a default model
    parser = argparse.ArgumentParser(description='Train a model.')
    parser.add_argument('--model', type=str, choices=['ElasticNet', 'Ridge', 'XGBRegressor'], default='XGBRegressor', help='The model to train. Defaults to XGBRegressor.')
    args = parser.parse_args()

    train(args.model)

```

#### MLproject file and Running Locally

The `MLproject` file contains only one entry point:

```yaml
name: "Housing Price Prediction"

conda_env: conda.yaml

entry_points:
  main:
    parameters:
      model: {type: string, default: "ElasticNet", choices: ["ElasticNet", "Ridge", "XGBRegressor"]}
    command: "python train.py --model {model}"
```

Even though we could run the training via CLI, it is common to use the Python MLflow API, as done here with `run.py`:

```python
import mlflow

models = ["ElasticNet", "Ridge", "XGBRegressor"]
entry_point = "main"

# We will change this depending on local tests / AWS runs
#mlflow.set_tracking_uri("http://ec2-<IP-number>.<region>.amazonaws.com:5000")
mlflow.set_tracking_uri("http://127.0.0.1:5000")

for model in models:
    experiment_name = model
    mlflow.set_experiment(experiment_name)
    
    mlflow.projects.run(
        uri=".",
        entry_point=entry_point,
        parameters={"model": model},
        env_manager="conda"
    )
```

To use `run.py`:

```bash
# Terminal 1: Start MLflow Tracking Server
conda activate mlflow
cd .../mlflow-housing-price-example
mlflow server

# Terminal 2: Run pipeline
# Since we are running a hyperparameter tuning
# the execution might take some time
# WARNING: the XGBoostRegressor model has many parameter combinations!
conda activate mlflow
cd .../mlflow-housing-price-example
python run.py

# Browser: Open URI:5000 == http://127.0.0.1:5000
# We should see 3 (Ridge, ElasticNet, XGBoostRegressor) experiments with several runs each,
# containing metrics, parameters, artifacts (dataset & model), etc.
```

![MLflow UI: Local Runs](./assets/mlflow_local_runs_example.jpg)

After we have finished, we commit to the AWS CodeCommit repo.

### Setup AWS Sagemaker

We log in with the IAM user credentials.

AWS SageMaker is the service from AWS to do anything related to ML: from notebook to deployment.

To set Sagemaker up, we need to:

- Add our repository
- Create a new IAM role with permissions for: CodeCommit (repo), S3 (store), ECR (build container image for deployment)
  - Note: we have create an IAM user, but now we need an IAM role, these are different things.
- Create a notebook instance: **even though it's called notebook, it's really a Jupyter Lab server where we have Python scripts, a terminal and also notebooks; in fact we're going to use the Terminal along with scripts, no the notebooks**.

To add our repository to SageMaker:

    Search: SageMaker
    Left panel: Notebook > Git repositories
      Add repository
        We can choose AWS CodeCommit / Github / Other
        Choose AWS CodeCommit
        Select
          Our repo: mlflow-housing-price-example
          Branch: master
          Name: mlflow-housing-price-example
      Add repository

To create a new IAM role with the necessari permissions:

    Username (up left) > Security credentials
    Access management (left panel): (IAM) roles > Create role
      AWS Service
      Use case: SageMaker
        The role will have SageMaker execution abilities
      Next
      Name
        Role name: house-price-role
      Create role

    Now, we need to give it more permissions: CodeCommit, S3, ECR (to build image container)
    IAM Roles: Open role 'house-price-role'
      Permission policies: we should see AmazonSageMakerFullAccess
      Add permissions > Attach policies:
        AWSCodeCommitFullAccess
        AWSCodeCommitPowerUser
        AmazonS3FullAccess
        EC2InstanceProfileForImageBuilderECRContainerBuilds

To create a notebook instance on SageMaker:

    Left panel: Notebook > Notebook instances
      Create notebook instance
        Notebook instance settings
          Name: house-price-nb
          Instance type: ml.t3.large (smaller ones might fail)
        Permissions and encryption
          IAM role: house-price-role (just created)
          Enable root access to the notebook
        Git repositories: we add our repo
          Default repository: mlflow-housing-price-example - AWS CodeCommit
        Create notebook instance

Then, the notebook instance is created; we wait for it to be in service, then, we `Open JupyterLab`. We will see that the CodeCommit repository has been cloned to the Jupyter Lab file system: `/home/ec2-user/SageMaker/mlflow-housing-price-example`.

In the Jupyter Lab server instance, we can open

- Code files: however, we ususally develop on our local machine.
- Terminals: however, the conda.yaml environment is not really installed.
- Notebooks: we don't need to use them really, we can do everything with our Python files an the terminal.
- etc.

If we locally/remotely modify anything in the files, we can synchronize as usually with the repository; in a Terminal:

```bash
cd .../mlflow-housing-price-example
git pull
git push
```

Therefore, we can actually develop on our local machine and 

- `git push` on our machine
- `git pull` on a SageMaker notebook terminal

**If we stop the SageMaker Notebook instance to safe costs, we need to re-start it.** The notebook local data will persist, but if we have installed anything in the environment (e.g., mlflow), we need to re-install it.

### Training on AWS Sagemaker

We open the notebook instance on SageMaker:

    Left panel: Notebook > Notebook instances
      house-price-nb > Open JupyterLab

Then, in the Jupyter Lab instance:

- `git pull` in a Terminal, just in case
- Change the tracking server URI to the AWS EC2 public DNS in the `run.py` (we could use `dotenv` instead...)

    ```
    PREVIOUS: http://127.0.0.1:5000
    NEW: http://ec2-<IP-number>.<region>.amazonaws.com:5000 (Public IPv4 from the EC2 instance)
    ```
- Run the `run.py` script in the Termnial:

    ```bash
    # Go to the repo folder, if not in there
    cd .../mlflow-housing-price-example
    
    # We install mlflow just to be able to load MLproject and execute run.py
    # which will then set the correct conda environment in conda.yaml
    pip install mlflow

    # Run all experiments
    python run.py
    ```
- Open the MLflow UI hosted on AWS with the browser: `http://ec2-<IP-number>.<region>.amazonaws.com:5000`

Now, we can see that the entries that appear in the server/UI hosted on AWS are similar to the local ones; however:

- On AWS, if we pick a run and check the artifact URIs in it, we'll see they are of the form `s3://<bucket-name>/...`.
- If we go to the S3 UI on the AWS web interface and open the created bucket, we will see the artifacts.

### Model Comparison and Evaluation

Model comparison and evaluation is done as briefly introduced in [MLflow UI - 01_tracking](#mlflow-ui---01_tracking):

- We pick an experiment.
- We select the runs we want, e.g., all.
- We click on `Compare`.
- We can use either the plots or the metrics.
- We select the best one (e.g., the one with the best desired metric), open the Run ID and click on `Register model`.
  - Create a model, e.g., `best`.
    - Note: if we create one model registry and assign the different experiment models as versions to it we can compare the selected ones in the registry.
    - Alternative: we create a registry of each model: `elasticnet`, `ridge`, `xgboost`.
  - Go to the `Models` tab (main horizontal menu), click on a model version, add tags, e.g., `staging`.

![Comparing Runs: Plots](./assets/mlflow_comparing_runs_plot.jpg)
![Comparing Runs: Metrics](./assets/mlflow_comparing_runs_metrics.jpg)
![Comparing Runs: Registering](./assets/mlflow_register_ui.jpg)

Usually, we add the alias `production` to the final selected model.

To get the **model URI**, select the model version we want, go to the run link and find the artifact path, e.g.:

    s3://<bucket-name>/<experiment-id>/<run-id>/artifacts/XGBRegressor


### Deployment on AWS Sagemaker

In order to deploy a model, first we need to build a parametrized docker image with the command [`mlflow sagemaker build-and-push-container`](https://mlflow.org/docs/latest/cli.html#mlflow-sagemaker-build-and-push-container). This command doesn't directly deploy a model; instead, it prepares the necessary environment for model deployment, i.e., it sets the necessary dependencies in an image.

```bash
# Open a terminal in the SageMaker Jupyter Lab instance
cd .../mlflow-housing-price-example
# We should see the conda.yaml in here

# Build parametrized image and push it to ECR
# --container: image name
# --env-manager: local, virtualenv, conda
mlflow sagemaker build-and-push-container --container xgb --env-manager conda
```

We can check that the image is pushed to AWS ECR

    Search: ECR
    Private registry (left panel): Repositories - the image with the name in --container should be there
      We can copy the image URI, e.g.
      <ID>.ecr.<region>.amazonaws.com/<image-name>:<>

After the image is pushed, we deploy a container of it by running `deploy.py`, where many image parameters are defined:

- Bucket name
- Image URL
- Instance type
- Endpoint name
- Model URI
- ...

These parameters are passed to the function [create_deployment](https://mlflow.org/docs/latest/python_api/mlflow.sagemaker.html?highlight=create_deployment#mlflow.sagemaker.SageMakerDeploymentClient.create_deployment):

```python
import mlflow.sagemaker
from mlflow.deployments import get_deploy_client

# Specify a unique endpoint name (lower letters)
# FIXME: some the following should be (environment/repo) variables
endpoint_name = "house-price-prod"
# Get model URI from MLflow model registry (hosted on AWS)
model_uri = "s2://<bucket-name>/..."
# Get from IAM roles
execution_role_arn = "arn:aws:aim:..."
# Get from S3
bucket_name = "..."
# Get from ECR: complete image name with tag
image_url = "<ID>.ecr.eu-central.<region>.amazonaws.com/<image-name>:<tag>"
flavor = "python_function"

# Define the missing configuration parameters as a dictionary:
# region, instance type, instance count, etc.
config = {
    "execution_role_arn": execution_role_arn,
    "bucket_name": bucket_name,
    "image_url": image_url,
    "region_name": "eu-central-1", # usually, same as rest of services
    "archive": False,
    "instance_type": "ml.m5.xlarge", # https://aws.amazon.com/sagemaker/pricing/instance-types/
    "instance_count": 1,
    "synchronous": True
}

# Initialize a deployment client for SageMaker
client = get_deploy_client("sagemaker")

# Create the deployment
client.create_deployment(
    name=endpoint_name,
    model_uri=model_uri,
    flavor=flavor,
    config=config,
)
```

After we have modified manually the `deploy.py` in the AWS Jupyter Lab instance, we can run it on a Terminal:

```bash
# Go to folder
cd .../mlflow-housing-price-example

# Install mlflow, if not present (e.g., if the notebook was re-started)
pip install mlflow

# Deploy: This takes 5 minutes...
python deploy.py
```

When the deployment finished, we should see an endpoint entry in AWS:

    Search: Sagemaker
    Inference (left menu/panel) > Endpoints
      house-price-prod should be there

Now, the endpoint is up and running, ready to be used for inferences.

### Model Inference

Finally, we can run the inference, either locally (if the `.env` file is properly set) or remotely (in the Sagemaker notebook instance); this is the script `predict.py` for that:

```python
from data import test
import boto3
import json
from dotenv import load_dotenv

# Load environment variables in .env:
# AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION
load_dotenv()

# Defined in deploy.py
# FIXME: These values should be (environment/repo) variables
endpoint_name = "house-price-prod"
region = 'eu-central-1'

sm = boto3.client('sagemaker', region_name=region)
smrt = boto3.client('runtime.sagemaker', region_name=region)

test_data_json = json.dumps({'instances': test[:20].toarray()[:, :-1].tolist()})

prediction = smrt.invoke_endpoint(
    EndpointName=endpoint_name,
    Body=test_data_json,
    ContentType='application/json'
)

prediction = prediction['Body'].read().decode("ascii")

print(prediction)
```

### Clean Up

We need to remove these AWS services (in the region we have worked):

- [x] SageMaker Endpoint (Inference)
- [x] SageMaker Notebook
- [x] S3 bucket
- [x] ECR image (Repository)
- [x] EC2 instance (Terminate)

We can keep the role and keys.

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
