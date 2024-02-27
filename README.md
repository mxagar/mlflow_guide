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
  - [4. MLflow Logging Functions](#4-mlflow-logging-functions)
  - [5. Launch Multiple Experiments and Runs](#5-launch-multiple-experiments-and-runs)
  - [6. Autologging in MLflow](#6-autologging-in-mlflow)
  - [7. Tracking Server  of MLflow](#7-tracking-server--of-mlflow)
  - [8. MLflow Model Component](#8-mlflow-model-component)
  - [9. Handling Customized Models in MLflow](#9-handling-customized-models-in-mlflow)
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

See also: [MLflow Tracking Quickstart](https://mlflow.org/docs/latest/getting-started/intro-quickstart/index.html)

In the section example, a regularized linear regression is run using `ElasticNet` from `sklearn` (it combines L1 and L2 regularizations).

Summary of [`01_tracking/basic_regression_mlflow.py`](./examples/01_tracking/basic_regression_mlflow.py):

```python
# Imports
import mlflow
import mlflow.sklearn

# ...
# Create experiment, if not existent, else set it
exp = mlflow.set_experiment(experiment_name="experment_1")

# ...
# Run experiments in with context
with mlflow.start_run(experiment_id=exp.experiment_id):
    # Fit model
    lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
    lr.fit(train_x, train_y)
    
    # Predict and evaluate
    predicted_qualities = lr.predict(test_x)
    (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

    # Log: parameters, metrics, model itself
    mlflow.log_param("alpha", alpha)
    mlflow.log_param("l1_ratio", l1_ratio)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("mae", mae)
    mlflow.sklearn.log_model(lr, "mymodel") # dir name in the artifacts to dump model
```

We can run the script as follows:

```bash
cd examples/01_tracking
# Run 1
python ./basic_regression_mlflow.py # default parameters
# Run 2
python ./basic_regression_mlflow.py --alpha 0.5 --l1_ratio 0.1
# Run 3
python ./basic_regression_mlflow.py --alpha 0.1 --l1_ratio 0.9
```

Then, a folder `mlruns` is created, which contains all the information of the experiments we create and the associated runs we execute.

This `mlruns` folder is very important, and it contains the following

```
.trash/           # deleted infor of experiments, runs, etc.
0/                # default experiment, ignore it
99xxx/            # our experiment, hashed id
  meta.yaml       # experiment YAML: id, name, creation time, etc.
  8c3xxx/         # a run, for each run we get a folder with an id
    meta.yaml     # run YAML: id, name, experiment_id, time, etc.
    artifacts/
      mymodel/    # dumped model: PKL, MLmodel, conda.yaml, requirements.txt, etc.
        ...
    metrics/      # once ASCII file for each logged metric
    params/       # once ASCII file for each logged param
    tags/         # metadata tags, e.g.: run name, committ hash, filename, ...
  6bdxxx/         # another run
    ...
```

 Notes:

- We can specify where this `mlruns` folder is created.
- Usually, the `mlruns` folder should be in a remote server; if local, we should add it to `.gitignore`.
- Note that `artifacts/` contains everything necessary to re-create the environment and load the trained model!
- Usually, the UI is used to visualize the metrics; see below.


### MLflow UI - 01_tracking



## 4. MLflow Logging Functions

## 5. Launch Multiple Experiments and Runs

## 6. Autologging in MLflow

## 7. Tracking Server  of MLflow

## 8. MLflow Model Component

## 9. Handling Customized Models in MLflow

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
