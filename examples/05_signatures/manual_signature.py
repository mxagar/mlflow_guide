'''Basic Linear Regression, regularized with ElasticNet.

Original code from the Udemy course

    MLflow in Action - Master the art of MLOps using MLflow tool
    https://www.udemy.com/course/mlflow-course/

    Author: J Garg, Real Time Learning

Modified following https://mlflow.org/docs/latest/getting-started/intro-quickstart/index.html

Usage:

    conda activate <env_name>
    python ./<filename.py>

'''
import warnings
import argparse
import logging
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
import mlflow
import mlflow.sklearn
from pathlib import Path
import os
from mlflow.models.signature import ModelSignature, infer_signature
from mlflow.types.schema import Schema, ColSpec

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

# Get arguments from command
parser = argparse.ArgumentParser()
parser.add_argument("--alpha", type=float, required=False, default=0.7)
parser.add_argument("--l1_ratio", type=float, required=False, default=0.7)
args = parser.parse_args()

# Evaluation function
def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Read the wine-quality csv file from the URL
    data = pd.read_csv("../data/red-wine-quality.csv")
    Path("./data/").mkdir(parents=True, exist_ok=True)
    data.to_csv("data/red-wine-quality.csv", index=False)
    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)
    train.to_csv("data/train.csv")
    test.to_csv("data/test.csv")
    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    alpha = args.alpha
    l1_ratio = args.l1_ratio

    mlflow.set_tracking_uri(uri="")

    print("The set tracking uri is ", mlflow.get_tracking_uri())
    exp = mlflow.set_experiment(experiment_name="experiment_signature")
    #get_exp = mlflow.get_experiment(exp_id)

    print("Name: {}".format(exp.name))
    print("Experiment_id: {}".format(exp.experiment_id))
    print("Artifact Location: {}".format(exp.artifact_location))
    print("Tags: {}".format(exp.tags))
    print("Lifecycle_stage: {}".format(exp.lifecycle_stage))
    print("Creation timestamp: {}".format(exp.creation_time))

    mlflow.start_run()
    tags = {
        "engineering": "ML platform",
        "release.candidate":"RC1",
        "release.version": "2.0"
    }


    mlflow.set_tags(tags)
    mlflow.sklearn.autolog(
        log_input_examples=False,
        log_model_signatures=False,
        log_models=False
    )
    lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
    lr.fit(train_x, train_y)

    predicted_qualities = lr.predict(test_x)

    (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

    print("Elasticnet model (alpha={:f}, l1_ratio={:f}):".format(alpha, l1_ratio))
    print("  RMSE: %s" % rmse)
    print("  MAE: %s" % mae)
    print("  R2: %s" % r2)

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
    mlflow.log_artifact("data/red-wine-quality.csv")
    mlflow.sklearn.log_model(lr, "model", signature=signature, input_example=input_example)
    artifacts_uri=mlflow.get_artifact_uri()
    print("The artifact path is",artifacts_uri )
    mlflow.end_run()
    run = mlflow.last_active_run()
    print("Active run id is {}".format(run.info.run_id))
    print("Active run name is {}".format(run.info.run_name))
