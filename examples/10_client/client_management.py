'''Basic MLflow client calls.

Original code from the Udemy course

    MLflow in Action - Master the art of MLOps using MLflow tool
    https://www.udemy.com/course/mlflow-course/

    Author: J Garg, Real Time Learning

Modified following https://mlflow.org/docs/latest/getting-started/intro-quickstart/index.html

More information on mlflow.client.MlflowClient, used here:
    
    https://www.mlflow.org/docs/latest/python_api/mlflow.client.html?highlight=client#module-mlflow.client

Summary of tasks carried out:

- Experiments: creating, adding tags, renaming, getting and searching experiments, deleting, restoring.
- Runs: creating, renaming, settng status getting and searching runs, deleting, restoring.
- Logging/extracting parameters, metrics and artifacts via the client.
- Creating and registering model versions, setting tags, searching and getting models, deleting.

Usage:

    # Terminal 1
    conda activate <env_name>
    mlflow server # http://127.0.0.1:5000
    # Terminal 2
    conda activate <env_name>
    python ./<filename.py>

'''
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
