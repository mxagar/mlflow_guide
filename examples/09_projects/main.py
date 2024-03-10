'''Basic Linear Regression, regularized with ElasticNet.

Original code from the Udemy course

    MLflow in Action - Master the art of MLOps using MLflow tool
    https://www.udemy.com/course/mlflow-course/

    Author: J Garg, Real Time Learning

Modified following https://mlflow.org/docs/latest/getting-started/intro-quickstart/index.html

Usage:

    conda activate <env_name>
    python ./<filename.py> [--alpha 0.3] [--l1_ratio 0.3]

'''
import os
import warnings
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse
from pathlib import Path
import mlflow
import mlflow.sklearn
import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, required=False, default=0.4)
    parser.add_argument("--l1_ratio", type=float, required=False, default=0.4)
    args = parser.parse_args()

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

    mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")

    experiment = mlflow.set_experiment( experiment_name="Project exp 1" )

    print("Name: {}".format(experiment.name))
    print("Experiment_id: {}".format(experiment.experiment_id))

    with mlflow.start_run( experiment_id=experiment.experiment_id):

        run = mlflow.last_active_run()

        print("Active run_id: {}".format(run.info.run_id))
        print("Active run name: {}".format(run.info.run_name))

        lr = ElasticNet(alpha=args.alpha, l1_ratio=args.l1_ratio, random_state=42)
        lr.fit(train_x, train_y)

        predicted_qualities = lr.predict(test_x)

        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        print("Elasticnet model (alpha={:f}, l1_ratio={:f}):".format(args.alpha, args.alpha))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        mlflow.log_metrics({
            "rmse": rmse,
            "r2": r2,
            "mae": mae
        })

        mlflow.log_params({
            "alpha": args.alpha,
            "l1 ratio": args.l1_ratio
        })

        mlflow.sklearn.log_model(lr, "model")

if __name__ == "__main__":
    main()
