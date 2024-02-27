'''Basic Linear Regression, regularized with ElasticNet.

Original code from the Udemy course

    MLflow in Action - Master the art of MLOps using MLflow tool
    https://www.udemy.com/course/mlflow-course/

    Author: J Garg, Real Time Learning

Modified following https://mlflow.org/docs/latest/getting-started/intro-quickstart/index.html
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
from mlflow.models import infer_signature

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

#get arguments from command
parser = argparse.ArgumentParser()
parser.add_argument("--alpha", type=float, required=False, default=0.5)
parser.add_argument("--l1_ratio", type=float, required=False, default=0.5)
args = parser.parse_args()

#evaluation function
def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Read the wine-quality csv file from local
    data = pd.read_csv("../data/red-wine-quality.csv")
    data.to_csv("../data/red-wine-quality.csv", index=False)

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)

    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    alpha = args.alpha
    l1_ratio = args.l1_ratio
        
    # It is recommended to fit and evaluate the model outside
    # of the `with` context in which the run is logged:
    # in case something goes wrong, no run is created
    lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
    lr.fit(train_x, train_y)
    # Predict and evaluate
    predicted_qualities = lr.predict(test_x)
    (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

    print("Elasticnet model (alpha={:f}, l1_ratio={:f}):".format(alpha, l1_ratio))
    print("  RMSE: %s" % rmse)
    print("  MAE: %s" % mae)
    print("  R2: %s" % r2)
    
    # We set the empty URI
    mlflow.set_tracking_uri(uri="my_tracking")
    # We get the URI
    print("The set tracking uri is ", mlflow.get_tracking_uri())
    # Create experiment, if not existent, else set it
    exp = mlflow.set_experiment(experiment_name="experment_2")

    # Infer the model signature
    signature = infer_signature(train_x, lr.predict(train_x))

    # Log run in `with` context
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
