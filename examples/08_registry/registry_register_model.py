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

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

# Get arguments from command
parser = argparse.ArgumentParser()
parser.add_argument("--alpha", type=float, required=False, default=0.6)
parser.add_argument("--l1_ratio", type=float, required=False, default=0.6)
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
    # Data artifacts
    train, test = train_test_split(data)
    data_dir = 'data'
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    data.to_csv(data_dir + '/dataset.csv')
    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)

    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    alpha = args.alpha
    l1_ratio = args.l1_ratio

    # Connect to the tracking server
    # Make sure a server is started with the given URI: mlflow server ...
    mlflow.set_tracking_uri(uri="http://127.0.0.1:5000/")

    print("The set tracking uri is ", mlflow.get_tracking_uri())
    exp = mlflow.set_experiment(experiment_name="experiment_register_model_api")

    print("Name: {}".format(exp.name))
    print("Experiment_id: {}".format(exp.experiment_id))
    print("Artifact Location: {}".format(exp.artifact_location))
    print("Tags: {}".format(exp.tags))
    print("Lifecycle_stage: {}".format(exp.lifecycle_stage))
    print("Creation timestamp: {}".format(exp.creation_time))

    mlflow.start_run()
    tags = {
        "engineering": "ML platform",
        "release.candidate": "RC1",
        "release.version": "2.0"
    }

    mlflow.set_tags(tags)

    lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
    lr.fit(train_x, train_y)

    predicted_qualities = lr.predict(test_x)

    (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

    print("Elasticnet model (alpha={:f}, l1_ratio={:f}):".format(alpha, l1_ratio))
    print("  RMSE: %s" % rmse)
    print("  MAE: %s" % mae)
    print("  R2: %s" % r2)

    mlflow.log_params({
        "alpha": 0.3,
        "l1_ratio": 0.3
    })

    mlflow.log_metrics({
        "rmse": rmse,
        "r2": r2,
        "mae": mae
    })
    run = mlflow.last_active_run()
    
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
    
    (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)
    print("  RMSE_test: %s" % rmse)
    print("  MAE_test: %s" % mae)
    print("  R2_test: %s" % r2)

    mlflow.end_run()
    run = mlflow.last_active_run()
